from __future__ import print_function, division
import random
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import timm
from torchvision import models as tvmodels
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import albumentations as A
import numpy as np
import cv2
from sklearn.model_selection import GroupKFold, StratifiedKFold
import data_loader.dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = '/data/wangshiyuan/data/cassava/'
NUM_FOLDS = 5
bs = 32
# Running only 5 epochs to test (Train more offline ^_^)
EPOCHS = 25
sz = 512
SNAPMIX_ALPHA = 5.0
SNAPMIX_PCT = 0.5
GRAD_ACCUM_STEPS = 1
TIMM_MODEL = 'resnet50'

eval()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 1234
seed_everything(SEED)
train_df = pd.read_csv('/data/fengjing/data/cassava/train_ori.csv')
DATA_PATH = '/data/fengjing/data/cassava/'
def train_transforms():
    return Compose([
            A.RandomResizedCrop(sz, sz),
            #A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def valid_transforms():
    return Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)



def accuracy_metric(input, targs):
    return accuracy_score(targs.cpu(), input.cpu())


def print_scores(scores):
    kaggle_metric = np.average(scores)
    print("Kaggle Metric: %f" % (kaggle_metric))

    return kaggle_metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def checkpoint(model, optimizer, epoch, current_metric, best_metric, fold):
    print("Metric improved from %f to %f , Saving Model at Epoch #%d" % (best_metric, current_metric, epoch))
    ckpt = {
        'model': CassavaNet(),
        'state_dict': model.state_dict(),
        #'optimizer' : optimizer.state_dict(),  # Commenting this out to cheap out on space
        'metric': current_metric
    }
    torch.save(ckpt, 'ckpt_%s-%d-%d.pth' % (TIMM_MODEL, sz, epoch))

folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True,
                        random_state=SEED).split(np.arange(train_df.shape[0]),
                                                 train_df.label.values)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_spm(input,target,model):
    imgsize = (sz, sz)
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        clsw = model.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i,target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps,clslogit


def snapmix(input, target, alpha, model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if True:
        wfmaps,_ = get_spm(input, target, model)
        bs = input.size(0)
        lam = np.random.beta(alpha, alpha)
        lam1 = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()


class SnapMixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss


for fold_num, (train_split, valid_split) in enumerate(folds):

    train_set = train_df.iloc[train_split].reset_index(drop=True)
    valid_set = train_df.iloc[valid_split].reset_index(drop=True)
    #     print(train_set)
    train_ds = CassavaDataset(dataframe=train_set,
                              root_dir=DATA_PATH + 'train_images',
                              transforms=train_transforms())

    valid_ds = CassavaDataset(dataframe=valid_set,
                              root_dir=DATA_PATH + 'train_images',
                              transforms=valid_transforms())

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs,
                                           shuffle=True, num_workers=8, drop_last=True,
                                           pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs,
                                           shuffle=False, num_workers=8,
                                           pin_memory=True)
    losses = []
    batches = len(train_dl)
    val_batches = len(valid_dl)
    best_metric = 0

    model = CassavaNet().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    snapmix_criterion = SnapMixLoss().to(device)
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': 1e-2},
        {'params': model.classifier.parameters()},
    ]
    optimizer = torch.optim.SGD(param_groups, lr=1e-1, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 20, 40],
                                                     gamma=0.1, last_epoch=-1, verbose=True)
    scaler = GradScaler()

    print("epochs:", EPOCHS)
    for epoch in range(EPOCHS):
        # ----------------- TRAINING  -----------------
        train_loss = 0
        progress = tqdm(enumerate(train_dl), desc="Loss: ", total=batches)

        model.train()
        for i, data in progress:
            image, label = data.values()
            #             print(data)
            #             print(image)
            #             print(label)
            #             label = torch.Tensor(label)
            X, y = image.to(device).float(), label.to(device).long()

            with autocast():

                rand = np.random.rand()
                if rand > (1.0 - SNAPMIX_PCT):
                    X, ya, yb, lam_a, lam_b = snapmix(X, y, SNAPMIX_ALPHA, model)
                    outputs, _ = model(X)
                    loss = snapmix_criterion(criterion, outputs, ya, yb, lam_a, lam_b)
                else:
                    outputs, _ = model(X)
                    loss = torch.mean(criterion(outputs, y))

            scaler.scale(loss).backward()
            # Accumulate gradients
            if ((i + 1) % GRAD_ACCUM_STEPS == 0) or ((i + 1) == len(train_dl)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            cur_step = i + 1
            trn_epoch_result = dict()
            trn_epoch_result['Epoch'] = epoch + 1
            trn_epoch_result['train_loss'] = round(train_loss / cur_step, 4)

            progress.set_description(str(trn_epoch_result))

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------- VALIDATION  -----------------
        val_loss = 0
        scores = []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_dl):
                image, label = data.values()
                X, y = image.to(device), label.to(device)
                outputs, _ = model(X)
                l = val_criterion(outputs, y)
                val_loss += l.item()

                preds = F.softmax(outputs).argmax(axis=1)
                scores.append(accuracy_metric(preds, y))

        epoch_result = dict()
        epoch_result['Epoch'] = epoch + 1
        epoch_result['train_loss'] = round(train_loss / batches, 4)
        epoch_result['val_loss'] = round(val_loss / val_batches, 4)

        print(epoch_result)

        # Check if we need to save
        current_metric = print_scores(scores)
        if current_metric > best_metric:
            checkpoint(model, optimizer, epoch + 1, current_metric, best_metric, fold_num)
            best_metric = current_metric

    del model, optimizer, train_dl, valid_dl, scaler, scheduler
    torch.cuda.empty_cache()

    # Train only a single fold
    break