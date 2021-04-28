import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import config
import random
class terrorismDataset(Dataset):
    """Cassava dataset."""

    def __init__(self,  data_list, transforms=None,out_put_label = True):
        """
        Args:
            dataframe (string): dataframe train/valid
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        # self.root_dir = root_dir
        self.transforms = transforms
        self.data_list = data_list
        self.path_memory ='/data/fengjing/terr_test/baokong_train/bloody_car/bloody_car (1792).jpg'
        self.out_put_label = out_put_label

    def __len__(self):
        return len(self.data_list)

    def get_img_bgr_to_rgb(self, path):

        try:
            im_bgr = cv2.imread(path)
            im_rgb = im_bgr[:, :, ::-1]
            self.path_memory = path
        except:
            path = self.path_memory
            im_bgr = cv2.imread(path)
            im_rgb = im_bgr[:, :, ::-1]
            # print('无效图片',path)
        return im_rgb

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.out_put_label:
            img_name,label  = self.data_list[idx]
        else:
            img_name = self.data_list[idx]
        image = self.get_img_bgr_to_rgb(img_name)
        if self.transforms:
            image = self.transforms(image=image)['image']

        if not self.out_put_label:
            sample = {
                'image': image,
            }
        else:
            sample = {
                'image': image,
                'label': int(label),
            }

        return sample



class Batch_Balanced_Dataset(object):
    def __init__(self, dataset_list: list, ratio_list: list, module_args: dict,
                 phase: str = 'train'):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
        :param dataset_list: 数据集列表
        :param ratio_list: 比例列表
        :param module_args: dataloader的配置
        :param phase: 训练集还是验证集
        """
        assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        all_batch_size = module_args['loader']['train_batch_size'] if phase == 'train' else module_args['loader'][
            'val_batch_size']
        for _dataset, batch_ratio_d in zip(dataset_list, ratio_list):
            _batch_size = max(round(all_batch_size * float(batch_ratio_d)), 1)

            _data_loader = DataLoader(dataset=_dataset,
                                      batch_size=_batch_size,
                                      shuffle=module_args['loader']['shuffle'],
                                      num_workers=module_args['loader']['num_workers'])

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_len += len(_dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        balanced_batch_images = []
        balanced_batch_score_maps = []
        balanced_batch_training_masks = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, score_map, training_mask = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_score_maps.append(score_map)
                balanced_batch_training_masks.append(training_mask)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, score_map, training_mask = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_score_maps.append(score_map)
                balanced_batch_training_masks.append(training_mask)
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_batch_score_maps = torch.cat(balanced_batch_score_maps, 0)
        balanced_batch_training_masks = torch.cat(balanced_batch_training_masks, 0)
        return balanced_batch_images, balanced_batch_score_maps, balanced_batch_training_masks
