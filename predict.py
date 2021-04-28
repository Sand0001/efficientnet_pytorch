import os
import cv2
import time
import torch
import numpy as np
from networks import get_model
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
import random
class Pytorch_model:


    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # config = checkpoint['config']
        # config['arch']['args']['pretrained'] = False
        # self.net = get_model(model_name='resnet50')
        # self.net = get_model(model_name='efficientnet-b3')
        self.net = get_model(model_name='tf_efficientnet_b2_ns')
        # self.net = get_model(model_name='efficientnet-b2')
        # self.net = get_model(model_name='resnext50_32x4d')

        # self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        # self.net.load_state_dict(checkpoint['state_dict'],False)
        self.net.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

        self.net.to(self.device)


        self.net.eval()
        self.img_channel = 3

    def valid_transforms(self,sz):
        return Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    def predict(self, img, size:224):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''


        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img,(size,size))
        h, w = img.shape[:2]
        # scale = short_size / min(h, w)
        # img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        # imgs = np.array([img,img])
        tensor = transforms.ToTensor()(img)

        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

        tensor = tensor.unsqueeze_(0)
        # tensor = torch.stack((tensor,tensor),0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            outputs, _ = self.net(tensor)
            preds = F.softmax(outputs)
            class_info = preds.argmax(dim=1)
            score = preds[0,class_info]
            return preds.cpu().numpy()
            # return class_info.cpu().numpy(),score.cpu().numpy()
    def predict2(self, img, size:224):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''


        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img,(size,size))
        h, w = img.shape[:2]
        # scale = short_size / min(h, w)
        # img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        # imgs = np.array([img,img])
        tensor = transforms.ToTensor()(img)

        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

        tensor = tensor.unsqueeze_(0)
        # tensor = torch.stack((tensor,tensor),0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            outputs, _ = self.net(tensor)
            preds = F.softmax(outputs)
            class_info = preds.argmax(dim=1)
            score = preds[0,class_info]
            # return preds.cpu().numpy(
            return class_info.cpu().numpy(),score.cpu().numpy()


    def predict_batch(self, imgs,pic_path, size: 224):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''

        for i,img in enumerate(imgs):
            img = os.path.join(pic_path,img)
            if os.path.exists(img):
                try:
                    img = cv2.imread(img)
                    if self.img_channel == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (size, size))
                    h, w = img.shape[:2]
                    if i==0:
                        tensor_final = transforms.ToTensor()(img)
                        tensor_final = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_final)

                        tensor_final = tensor_final.unsqueeze_(0)
                    else:
                        tensor = transforms.ToTensor()(img)
                        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

                        tensor = tensor.unsqueeze_(0)
                        try:
                            tensor_final  = torch.cat((tensor_final, tensor), 0)
                        except:
                            tensor_final = tensor
                            tensor_final  = torch.cat((tensor, tensor), 0)

                except:
                    continue

            else:

                print('file is not exists')
                continue

        tensor = tensor_final.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            outputs, _ = self.net(tensor)
            preds = F.softmax(outputs)
            # class_info = preds.argmax(dim=1)
            # score = preds[0,class_info]
            return preds.cpu().numpy()


            # if str(self.device).__contains__('cuda'):
            #     torch.cuda.synchronize(self.device)


def post_processing(preds):
    all_bloody_prob = 0
    all_carbloody_prob = 0
    all_normal_prob = 0
    all_flag_prob = 0
    all_emblem_prob = 0
    all_carcrash_prob = 0
    all_uniform_prob = 0
    all_gun_prob = 0
    all_knife_prob = 0
    all_money_prob = 0
    all_politics_pro = 0
    all_scene_prob = 0
    all_drug_prob = 0
    all_map_prob = 0
    for class_info in range(1,len(preds)):
        class_info_ori = class_info
        # if class_info ==15:
        #     print(class_info)
        tmp_name = config.num2name[class_info]
        if tmp_name in config.class_dict2:
            class_info = config.class_dict2[tmp_name]
        score = preds[class_info_ori]

        if class_info==0:
            all_carbloody_prob+=np.round(score*10000)/100
        elif  class_info==1:
            all_carcrash_prob+=np.round(score*10000)/100
        elif class_info==2 or class_info ==3 or class_info ==4:
            all_bloody_prob += np.round(score * 10000) / 100
        elif class_info==5 or class_info ==9 or class_info ==10:
            all_scene_prob += np.round(score * 10000) / 100             #1
        elif class_info == 6 or class_info == 20:
            all_money_prob += np.round(score * 10000) / 100        #1
        elif  class_info==7:
            all_drug_prob+= np.round(score * 10000) / 100      #1
        elif  class_info==13:
            all_map_prob+= np.round(score * 10000) / 100      #1
        elif class_info==12 or class_info ==15 or class_info ==16 or class_info ==18 or class_info ==19 or class_info ==22:
            all_flag_prob += np.round(score * 10000) / 100 #1
        elif class_info==24 or class_info ==26 or class_info ==25 or class_info ==28:
            all_gun_prob += np.round(score * 10000) / 100 #1

        elif  class_info==27:
            all_knife_prob+= np.round(score * 10000) / 100#1
        elif class_info==11 or class_info ==14 or class_info ==17 or class_info ==21:
            all_emblem_prob += np.round(score * 10000) / 100#1

        elif class_info == 8 or class_info == 23:
            all_uniform_prob += np.round(score * 10000) / 100#1
        else:
            all_normal_prob += np.round(score * 10000) / 100

    all_politics_pro = all_scene_prob + all_gun_prob + all_flag_prob + all_emblem_prob + all_uniform_prob + all_knife_prob + all_money_prob + all_drug_prob + all_map_prob

    class_list = [("bloody", all_bloody_prob),
("carbloody", all_carbloody_prob),
("carcarsh", all_carcrash_prob),
("gun", all_gun_prob),
("flag", all_flag_prob),
("emblem", all_emblem_prob),
("uniform", all_uniform_prob),
("knife", all_knife_prob),
("money", all_money_prob),
("normal", all_normal_prob),
("politics", all_politics_pro),
("scene", all_scene_prob),
("drug", all_drug_prob),
("map", all_map_prob),]

    score_list =[ all_bloody_prob,
 all_carbloody_prob,
 all_carcrash_prob,
 all_gun_prob,
 all_flag_prob,
 all_emblem_prob,
all_uniform_prob,
 all_knife_prob,
 all_money_prob,
 all_normal_prob,
 all_politics_pro,
 all_scene_prob,
 all_drug_prob,
 all_map_prob,]
    a = np.array(score_list)
    max_score = np.max(a)
    max_score_index = score_list.index(max_score
                                       )

    return class_list[max_score_index]



if __name__ == '__main__':
    video = False
    # model = Pytorch_model(model_path='../checkpoint_efficient/ckpt-40.pth', gpu_id=0)
    # model = Pytorch_model(model_path='../checkpoint_resnet50/ckpt-78.pth', gpu_id=0)

    # model = Pytorch_model(model_path='checkpoint_resnext50/ckpt-best.pth', gpu_id=0)
    # model = Pytorch_model(model_path='checkpoint_efficient_b2/checkpoint/ckpt-best.pth', gpu_id=0)
    # model = Pytorch_model(model_path='/data/fengjing/checkpoint_b3_add_100years/ckpt-best.pth', gpu_id=0)
    model = Pytorch_model(model_path='/data/fengjing/terr_ckpt/ckpt-24.pth', gpu_id=0)
    # recall_list = [0] * 9

    threshold_list = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
    # count_right = [0]*len(threshold_list)
    recall_list = [0]*len(threshold_list)
    all_num = 0
    if video:
        terr_class_path = '/data/fengjing/terr_test/涉政暴恐抽帧图片'  #黑样本
        # terr_class_path = '/data/fengjing/terr_test/所有白样本关键帧抽帧图片'  #白样本
        terr_class_list = os.listdir(terr_class_path)
        all_recall_list = [0]*len(threshold_list)
        for i in range(len(terr_class_list)):

            per_class_path = os.path.join(terr_class_path, terr_class_list[i])
            per_class_list = os.listdir(per_class_path)
            print(terr_class_list[i],len(per_class_list))
            per_class_recall_num = [0]*len(threshold_list)
            for j in range(len(per_class_list)):
                all_num+=1
                vidio_path = os.path.join(per_class_path, per_class_list[j])
                pic_list = os.listdir(vidio_path)       ## 每个video的图片
                count_video_right = [0] * len(threshold_list)  ##计算每个video 是否召回

                for k in range(len(pic_list)):
                    pic = os.path.join(vidio_path, pic_list[k])
                    img = cv2.imread(pic)
                    try:
                        preds = model.predict(img, 224)
                        class_info, score = post_processing(preds[0])

                        for n in range(len(threshold_list)):
                            if score >= threshold_list[n]*100 and 'normal' not in class_info:
                                count_video_right[n] =1

                        if count_video_right[-1] == 1:
                            break


                    except Exception as e:
                        print(e)
                        print(pic)
                        continue
                for n in range(len(threshold_list)):
                    per_class_recall_num[n] += count_video_right[n]
                    all_recall_list[n]+= count_video_right[n]

            print(per_class_recall_num)


        print(threshold_list)
        print('召回',np.array(all_recall_list)/all_num)

    else:

        # pic_path = '/data/fengjing/terr_test/4th_test_5w_white/'
        # pic_path = '/data/fengjing/terr_test/100nian_test'

        # pic_path = '/data/fengjing/terr_test/wangpu_dataset/datasets/baokong/white/5w_test_tmp_white2/'
        pic_path = '/data/fengjing/terr_test/baokong_bk/' #黑样本
        pic_list = os.listdir(pic_path)
        random.shuffle(pic_list)
        pic_list = pic_list

        batchsize = 96*3
        recall_num = 0
        wrong_num = 0
        import time
        aa = time.time()
        all_num = 0

        for pic in pic_list:
            # print(pic)
            print(recall_list, all_num)

            pic = os.path.join(pic_path,pic)
            assert os.path.exists(pic), 'file is not exists'

            img = cv2.imread(pic)
            try:
                # plt.figure('1')
                # plt.imshow(img[:,:,(2,1,0)])
                # plt.show()
                preds = model.predict(img, 224)
                # class_info, score = model.predict(img, 224)
                class_info, score = post_processing(preds[0])
                # class_name = key_list[class_info[0]-1]
                # if score > 0.9 and 'normal' not in class_name:
                if score > 90 and 'normal' not in class_info:

                    recall_num += 1

                # else:

                    print(class_info, score)

                if score > 90 and 'normal' not in class_info:

                    recall_list[0] += 1
                if score > 85 and 'normal' not in class_info:
                    recall_list[1] += 1
                if score > 80 and 'normal' not in class_info:
                    recall_list[2] += 1
                if score > 75 and 'normal' not in class_info:
                    recall_list[3] += 1
                if score > 70 and 'normal' not in class_info:
                    recall_list[4] += 1
                if score > 65 and 'normal' not in class_info:
                    recall_list[5] += 1
                if score > 60 and 'normal' not in class_info:
                    recall_list[6] += 1
                if score > 55 and 'normal' not in class_info:
                    recall_list[7] += 1
                if score > 50 and 'normal' not in class_info:
                    recall_list[8] += 1
                all_num+=1
            except Exception as e:
                wrong_num+=1
                print(e)
                print(pic)
        print('haoshi ',time.time()-aa)
        print(all_num)
        print(recall_num,len(pic_list)-wrong_num)
        print(recall_list,len(pic_list)-wrong_num)
        print(np.array(recall_list)*1./all_num)

