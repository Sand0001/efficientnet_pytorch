from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F

import numpy as np
import logging
import time
import config
sz = 224
import json


def print_execute_time(func):
    from time import time

    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间和结束时间，将func夹在中间执行，取得其返回值
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        # 打印方法名称和其执行时间
        logging.info(f'{func.__name__}() execute time: {end - start}s')
        # 返回func的返回值
        return func_return

    # 返回嵌套的函数
    return wrapper
@print_execute_time
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
    all_100years_prob = 0
    for class_info in range(1, len(preds)):
        class_info_ori = class_info
        # if class_info ==15:
        #     print(class_info)
        tmp_name = config.num2name[class_info]
        if tmp_name in config.class_dict2:
            class_info = config.class_dict2[tmp_name]
        score = preds[class_info_ori]

        if class_info == 0:
            all_carbloody_prob += np.round(score * 10000) / 100
        elif class_info == 1:
            all_carcrash_prob += np.round(score * 10000) / 100
        elif class_info == 2 or class_info == 3 or class_info == 4:
            all_bloody_prob += np.round(score * 10000) / 100
        elif class_info == 5 or class_info == 9 or class_info == 10:
            all_scene_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 6 or class_info == 20:
            all_money_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 7:
            all_drug_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 13:
            all_map_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 12 or class_info == 15 or class_info == 16 or class_info == 18 or class_info == 19 or class_info == 22 :
            all_flag_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 24 or class_info == 26 or class_info == 25 or class_info == 28:
            all_gun_prob += np.round(score * 10000) / 100  # 1

        elif class_info == 27:
            all_knife_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 11 or class_info == 14 or class_info == 17 or class_info == 21:
            all_emblem_prob += np.round(score * 10000) / 100  # 1

        elif class_info == 8 or class_info == 23:
            all_uniform_prob += np.round(score * 10000) / 100  # 1
        elif class_info == 34:
            all_100years_prob = np.round(score * 10000) / 100

        else:
            all_normal_prob += np.round(score * 10000) / 100

    all_politics_pro = all_scene_prob + all_gun_prob + all_flag_prob + all_emblem_prob + \
                       all_uniform_prob + all_knife_prob + all_money_prob + all_drug_prob + all_map_prob+all_100years_prob



    result_class_dict = {"bloody" : all_bloody_prob,
                  "carbloody" : all_carbloody_prob,
                  "carcarsh" : all_carcrash_prob,
                  "gun" : all_gun_prob,
                  "flag" : all_flag_prob,
                  "emblem" : all_emblem_prob,
                  "uniform" : all_uniform_prob,
                  "knife" : all_knife_prob,
                  "money" : all_money_prob,
                  "normal" : all_normal_prob,
                  "politics" : all_politics_pro,
                  "scene" : all_scene_prob,
                  "drug" : all_drug_prob,
                  "map" : all_map_prob,
                    "100years" :all_100years_prob}

    return result_class_dict
@print_execute_time
def valid_transforms(sz):
    return Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
@print_execute_time
def get_img_bgr_to_rgb(image_bytes):
    im_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]

    return im_rgb
trans = valid_transforms(sz=sz)

@print_execute_time
def preprocess(image_bytes, **kwargs):

    # pic_path = '/data/fengjing/terr_test/baokong_bk/997_132.jpg'
    img = get_img_bgr_to_rgb(image_bytes)
    img = trans(image = img)['image']
    img = img.float().unsqueeze(0)
    print(img.shape)
    return img
@print_execute_time
def postprocess(outputs_results, **kwargs):
    logging.info('post start:'+str(time.time()))
    outputs,_= outputs_results
    preds = F.softmax(outputs)
    preds_numpy = preds.cpu().numpy()
    result_class_dict = post_processing(preds_numpy[0])
    result_class_dict = json.dumps(result_class_dict)

    logging.info('post end:'+str(time.time()))
    logging.info(result_class_dict)
    return result_class_dict

# if __name__ == '__main__':

