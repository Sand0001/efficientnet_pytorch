from . import dataset
import torch
from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold
import config
import random
import os
import numpy as np
def get_dataset(data_list,transform,out_put_label = True):
    """
    获取训练dataset
    :param data_list: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    s_dataset = getattr(dataset, 'terrorismDataset')(data_list=data_list,transforms=transform,out_put_label = out_put_label)
    return s_dataset


def train_transforms(sz):
    return Compose([
            A.RandomResizedCrop(sz, sz),
            #A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def load_data(data_list,root_dir) :
    print('Start loading data!')
    t_img_list = []
    t_label_list = []
    for file_path in data_list:

        pic_path = os.path.join(root_dir, file_path)
        label = config.class_dict[file_path]
        pic_list = os.listdir(pic_path)
        for pic in pic_list:
            t_img_list.append((os.path.join(pic_path, pic),label))
    random.shuffle(t_img_list)
    t_img_list = np.array(t_img_list)

    return t_img_list

def valid_transforms(sz):
    return Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_dataloader(data_list,root_dir,bs,sz):

    img_list = load_data(data_list,root_dir)
    folds = list(StratifiedKFold(n_splits=10, shuffle=True,random_state=1).split(img_list[:,0],img_list[:,1]))
    train_set = None
    val_set = None
    for (train_index,test_index) in folds:
        train_set = img_list[train_index]
        val_set = img_list[test_index]
        break
    print('训练集图片{}张，验证集图片{}张'.format(len(train_set),len(val_set)))
    train_ds = get_dataset(train_set,transform=train_transforms(sz))
    val_ds = get_dataset(val_set,transform=valid_transforms(sz))




    train_dl = DataLoader(train_ds, batch_size=bs,
                                           shuffle=True, num_workers=config.num_workers, drop_last=True,
                                           pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs,
                          shuffle=True, num_workers=config.num_workers, drop_last=True,
                          pin_memory=True)
    return train_dl,val_dl
