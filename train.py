from networks import get_model,get_loss
from data_loader import get_dataloader
from trainer import Trainer
import config
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
# GPU_ID_LIST = '2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID_LIST


def main():

    train_data,val_data = get_dataloader(data_list=config.data_list,root_dir=config.root_dir,bs=config.batchsize,sz=config.size)
    val_criterion = get_loss()
    criterion = get_loss(training=True)
    snapmix_criterion = get_loss(tag='snapmix')
    model = get_model(model_name='efficientnet-b3')
    # model = get_model(model_name='tf_efficientnet_b2_ns')
    # model = get_model(model_name='resnext50_32x4d')
    trainer = Trainer(config, model, criterion,val_criterion, snapmix_criterion,train_data,val_data,)
    trainer.train()
if __name__ == '__main__':
    main()