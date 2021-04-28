import config
from tqdm import tqdm
import os
import torch
# from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np
from snapmix import snapmix
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn

class Trainer():
    def __init__(self,config, model, criterion,val_criterion, snapmix_criterion,train_loader,val_loader, weights_init=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batches = len(train_loader)
        if val_loader!= None:
            self.val_batches = len(self.val_loader)
        else:
            self.val_batches = 0
        self.model = model
        self.model_save = model
        self.start_epoch = 0
        self.epochs = config.epochs
        self.device = torch.device("cuda:0")


        self.SNAPMIX_ALPHA = 5.0
        self.SNAPMIX_PCT = 0.5
        self.GRAD_ACCUM_STEPS = 1
        # self.scaler = GradScaler()
        self.best_metric = 0
        self.param_groups = [
            {'params': model.backbone.parameters(), 'lr': 1e-2},
            {'params': model.classifier.parameters()},
        ]

        # device
        torch.manual_seed(self.config.seed)  # 为CPU设置随机种子
        if len(self.config.gpus) > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            print(
                'train with gpu {} and pytorch {}'.format(self.config.gpus, torch.__version__))
            self.gpus = {i: item for i, item in enumerate(self.config.gpus)}
            self.device = torch.device("cuda:0")
            torch.cuda.manual_seed(self.config.seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config.seed)  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            print('train with cpu and pytorch {}'.format(torch.__version__))
            self.device = torch.device("cpu")
        self.criterion = criterion.to(self.device)
        self.val_criterion = val_criterion.to(self.device)
        self.snapmix_criterion = snapmix_criterion.to(self.device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model,device_ids=[0,1])
        self.model.to(self.device)

        self.warm_up_epochs = 5

        self.factor = 0.2
        self.patience = 5
        self.eps = 1e-6

        if config.optimizer == 'SGD':

            self.optimizer = torch.optim.SGD(self.param_groups, lr=1e-1, momentum=0.9,
                                    weight_decay=1e-4, nesterov=True)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 20, 40],
                                                                  gamma=0.1, last_epoch=-1)
        else:
            self.optimizer = torch.optim.Adam(self.param_groups, lr=1e-4,
                                             weight_decay=1e-6, amsgrad=False)
            self.scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor, patience=self.patience,
                                          verbose=True,
                                          eps=self.eps)

        # self.load_pretrained_model()
        # if config.warm_up == 'multistep_lr':
        #     # warm_up_with_multistep_lr
        #     warm_up_with_multistep_lr = lambda \
        #     epoch: epoch / self.warm_up_epochs if epoch <= self.warm_up_epochs else 0.1 ** len(
        #     [m for m in self.milestones if m <= epoch])
        #     self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_multistep_lr)
        # elif config.warm_up == 'cosine_lr':
        #
        #     # warm_up_with_cosine_lr
        #     warm_up_with_cosine_lr = lambda epoch: epoch / self.warm_up_epochs if epoch <= self.warm_up_epochs else 0.5 * (
        #                 math.cos((epoch - self.warm_up_epochs) / (self.epochs - self.warm_up_epochs) * math.pi) + 1)
        #     self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        #

    def load_pretrained_model(self):
        model_path = "/.cache/torch/checkpoints/efficientnet-b3-5fb5a3c3.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

    def _train_epoch(self,epoch):
        self.model.train()

        train_loss = 0
        progress = tqdm(enumerate(self.train_loader), desc="Loss: ", total=self.batches)

        for i, data in progress:
            image,label = data.values()

            # X, y = image.to(self.device).float(), label.to(self.device).long()
            X, y = image.to(self.device).float(), label.to(self.device).long()
            # with autocast():

            rand = np.random.rand()
            # print(self.model)
            if rand > (1.0 - self.SNAPMIX_PCT):
                X, ya, yb, lam_a, lam_b = snapmix(X, y, self.SNAPMIX_ALPHA, self.model)
                outputs, _ = self.model(X)
                loss = self.snapmix_criterion(self.criterion, outputs, ya, yb, lam_a, lam_b)
            else:
                outputs, _ = self.model(X)
                loss = torch.mean(self.criterion(outputs, y))

                # backward
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()
            # Accumulate gradients
            # if ((i + 1) % self.GRAD_ACCUM_STEPS == 0) or ((i + 1) == len(self.train_loader)):
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            #     self.optimizer.zero_grad()

            train_loss += loss.item()
            cur_step = i + 1
            trn_epoch_result = dict()
            trn_epoch_result['Epoch'] = epoch + 1
            trn_epoch_result['train_loss'] = round(train_loss / cur_step, 4)
            progress.set_description(str(trn_epoch_result))
        self.scheduler.step(train_loss)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('train_loss',train_loss/self.batches)



    def print_scores(self,scores):
        metric = np.average(scores)
        print("Metric: %f" % (metric))

        return metric

    def checkpoint(self,model,  epoch,current_metric):
        # print("Metric improved from %f to %f , Saving Model at Epoch #%d" % (best_metric, current_metric, epoch))
        ckpt = {
            'model': self.model_save,
            'state_dict': model.state_dict(),
            # 'optimizer' : optimizer.state_dict(),  # Commenting this out to cheap out on space
            # 'metric': current_metric
        }
        if not os.path.exists(config.checkpoint_dir):
            os.mkdir(config.checkpoint_dir)
        torch.save({'state_dict': model.state_dict()}, os.path.join(config.checkpoint_dir,'ckpt-%d.pth' % ( epoch)))
        if current_metric > self.best_metric:

            torch.save({'state_dict': model.state_dict()}, os.path.join(config.checkpoint_dir,'ckpt-best.pth' ))
            self.best_metric = current_metric

    def accuracy_metric(self,input, targs):
        return accuracy_score(targs.cpu(), input.cpu())

    def _log_memory_usage(self):
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        print(content)

    def eval(self,epoch):
        if self.val_loader!=None:
            # ----------------- VALIDATION  -----------------
            val_loss = 0.
            scores = []

            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    image, label = data.values()
                    X, y = image.to(self.device), label.to(self.device)
                    outputs, _ = self.model(X)
                    l = self.val_criterion(outputs, y)
                    val_loss += l.item()

                    preds = F.softmax(outputs).argmax(axis=1)
                    scores.append(self.accuracy_metric(preds, y))

            # epoch_result = dict()
            # epoch_result['Epoch'] = epoch + 1
            # epoch_result['train_loss'] = round(train_loss / self.batches, 4)
            print(val_loss)
            val_loss= round(val_loss / self.val_batches,4)

            print("val_loss ",val_loss)

            # Check if we need to save
            current_metric = self.print_scores(scores)
            self.checkpoint(self.model, epoch + 1,current_metric)

        else:
            pass



    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:

                epoch_train_loss = self._train_epoch(epoch)
                self.eval(epoch)


                # self.checkpoint(self.model, epoch + 1)

                # if self.config['lr_scheduler']['type'] != 'PolynomialLR':
                #     self.scheduler.step()
                # self._on_epoch_finish()
            except torch.cuda.CudaError:
                print(torch.cuda.CudaError)
                self._log_memory_usage()


        torch.cuda.empty_cache()




