# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import random
import math

from PIL import Image
from model_service.pytorch_model_service import PTServingBaseService

import numpy as np
import scipy.signal as signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class_n = 45
img_w, img_h = 64, 64
index_to_char = {0:"深", 1:"秦", 2:"京", 3:"海", 4:"成", 5:"南", 6:"杭", 7:"苏", 8:"松"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def diff(val_x, val_y):
    # 计算欧式距离
    diff_sum = 0
    for i in range(len(val_x)):
        diff_sum += (val_x[i] - val_y[i]) * (val_x[i] - val_y[i])
    return diff_sum

def reverse_write_and_black(flag):
    #翻转flag中的0和255
    res = np.array(flag)
    h, w = res.shape
    for i in range(h):
        for j in range(w):
            if res[i,j] == 255:
                res[i,j] = 0
            else:
                res[i,j] = 255
    return res

def get_pts(end_point, box_point):
    #判断应该选取哪三个点进行仿射变换
    res_index = []
    left_diff = diff(end_point[0], end_point[1])
    right_diff = diff(end_point[2], end_point[3])
    if left_diff > right_diff:
        res_index.extend([0, 1])
        if end_point[2, 1] - box_point[2, 1] > box_point[3, 1] - end_point[3, 1]:
            res_index.extend([2])
        else:
            res_index.extend([3])
    else:
        res_index.extend([2, 3])
        if end_point[0, 1] - box_point[0, 1] > box_point[1, 1] - end_point[1, 1]:
            res_index.extend([0])
        else:
            res_index.extend([1])
    return np.float32(end_point[res_index]), np.float32(box_point[res_index])

def my_preprocess(img):
    diff_threshold = 80
    search_step = 10
    point_width = 2
    flag_dst_threshold = 15
    
    img = img.copy()
    img = np.array(img)
    for i in range(len(img[0,0])):
        img[:,:,i] = signal.medfilt2d(img[:,:,i], kernel_size=5)
    h, w, _ = img.shape
    # 找出图像中拥有最多相似（欧式距离小于diff_threshold）的颜色，并认为这个颜色是车牌的底色
    count = np.zeros((256, 256, 256))
    for i in range(h):
        for j in range(w):
            count[img[i, j, 0], img[i, j, 1], img[i, j, 2]] += 1
    max_count = 0
    max_color = np.array([0, 0, 0])
    for i in range(0, h, h // search_step):
        for j in range(0, w, w // search_step):
            sum_count = 0
            sum_diff = 0
            for r in range(max(img[i, j, 0] - 10, 0), min(img[i, j, 0] + 10, 256)):
                r_diff = (r - img[i, j, 0]) * (r - img[i, j, 0])
                if r_diff < diff_threshold:
                    for g in range(max(img[i, j, 1] - 10, 0), min(img[i, j, 1] + 10, 256)):
                        g_diff = (g - img[i, j, 1]) * (g - img[i, j, 1])
                        if r_diff + g_diff < diff_threshold:
                            for b in range(max(img[i, j, 2] - 10, 0), min(img[i, j, 2] + 10, 256)):
                                b_diff = (b - img[i, j, 2]) * (b - img[i, j, 2])
                                if r_diff + g_diff + b_diff < diff_threshold:
                                    sum_count += count[r, g, b]
            if sum_count > max_count:
                max_count = sum_count
                max_color = img[i, j]
    flag = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if diff(img[i, j], max_color) < diff_threshold:
                flag[i, j] = 255
    # 找出车牌的四个端点
    flag_count = np.zeros((h, w))
    for i in range(point_width, h - point_width):
        for j in range(point_width, w - point_width):
            for x in range(i - point_width, i + point_width + 1):
                for y in range(j - point_width, j + point_width + 1):
                    if flag[x, y] == 255:
                        flag_count[i, j] += 1
            if flag_count[i, j] > (1 + point_width * 2) * (1 + point_width * 2) - 1:
                flag_count[i, j] = 255

    x_add_y_min = [1e6, 0, 0]
    x_add_y_max = [-1e6, 0, 0]
    x_sub_y_min = [1e6, 0, 0]
    x_sub_y_max = [-1e6, 0, 0]
    for x in range(h):
        for y in range(w):
            if flag_count[x, y] == 255:
                x_add_y = x + y
                x_sub_y = x - y
                if x_add_y < x_add_y_min[0]:
                    x_add_y_min = [x_add_y, x, y]
                if x_add_y > x_add_y_max[0]:
                    x_add_y_max = [x_add_y, x, y]
                if x_sub_y < x_sub_y_min[0]:
                    x_sub_y_min = [x_sub_y, x, y]
                if x_sub_y > x_sub_y_max[0]:
                    x_sub_y_max = [x_sub_y, x, y]
    # 判断需要选取车牌的哪三个端点进行仿射变换来修正车牌的位置
    # 左上，左下，右上，右下
    end_point = np.array([[x_add_y_min[2], x_add_y_min[1]], [x_sub_y_max[2], x_sub_y_max[1]], [x_sub_y_min[2], x_sub_y_min[1]], [x_add_y_max[2], x_add_y_max[1]]])
    box_point = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])

    pts1, pts2 = get_pts(end_point, box_point)

    M = cv2.getAffineTransform(pts1, pts2)
    flag_dst = cv2.warpAffine(reverse_write_and_black(flag), M, (w, h))
    img_dst = cv2.warpAffine(img, M, (w, h))
    # 对修正后的车牌进行划分，划分出9个字符
    char_segmentation = []
    img_dst_segment = img_dst.copy()
    flag_dst_count = [0] * w
    for x in range(h):
        for y in range(w):
            if flag_dst[x, y] == 255:
                flag_dst_count[y] += 1
    point_pos = w * 2 // 9

    l, r = 0, point_pos
    move_threshold = (r - l) // 10
    count = 0
    while l < r and flag_dst_count[l] < flag_dst_threshold and count < move_threshold:
        l += 1
        count += 1
    count = 0
    while r > l and  flag_dst_count[r] < flag_dst_threshold and count < move_threshold:
        r -= 1
        count += 1
    char_segmentation.append([l, (l + r) // 2])
    char_segmentation.append([(l + r) // 2, r])

    l, r = point_pos, w - 1
    move_threshold = (r - l) // 10
    count = 0
    while l < r and flag_dst_count[l] < flag_dst_threshold and count < move_threshold:
        l += 1
        count += 1
    count = 0
    while r > l and  flag_dst_count[r] < flag_dst_threshold and count < move_threshold:
        r -= 1
        count += 1
    
    count = 0
    for i in range(l, r + 1):
        if flag_dst_count[i] < flag_dst_threshold:
            count += 1
    count //= 12
    l = max(0, l - count)
    r = min(w - 1, r + count)
    l_pos = l
    for i in range(1, 7):
        r_pos = l + (r - l) * i // 7
        char_segmentation.append([l_pos, r_pos])
        l_pos = r_pos
    r_pos = r
    char_segmentation.append([l_pos, r_pos])
    return pts1, pts2, char_segmentation

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.classify = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 45),
            )

    def forward(self, x):
        x = self.feature(x)
        adaptiveAvgPoolWidth = x.shape[2]
        x = F.avg_pool2d(x, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

def getnet():
    model_ft = Net()
    model_ft.to(device)
    return model_ft

class PTVisionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        
        self.model_name = model_name
        self.model = getnet()
        print("model_name = ", model_name, " model_path = ", model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
        self.model.eval()

    def _preprocess(self, data):
        # data为文件流，单张图片使用data[0]
        #print("==================================")
        #print("type = ", type(data))
        #print(data)
        char_img_list = []
        for k, v in data.items():
            #print("k = ", k, " v = ", v)
            for file_name, img_path in v.items():
                #print("file_name = ",file_name ," img_path = ", img_path)
                img = Image.open(img_path)
                #print("start my preprocess")
                pts1, pts2, char_segmentation = my_preprocess(img)
                print("char_segmentation = ", char_segmentation)
                #print("finish my preprocess")
                img = np.array(img)
                h, w, _ = img.shape
                #print("img.shape = ", img.shape)

                M = cv2.getAffineTransform(pts1, pts2)
                img_dst = cv2.warpAffine(img, M, (w, h))

                for [x, y] in char_segmentation:
                    #print("x = ", x, " y = ", y)
                    #print("img_dst.shape = ", img_dst.shape)
                    char_img = cv2.resize(img_dst[:, x:y, :], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                    char_img_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(char_img)
                    char_img_tensor = char_img_tensor.to(device)
                    char_img_list.append(char_img_tensor)
                break
            break
        #print("char_img_list = ", char_img_list)
        print("************************************")
        return torch.stack(char_img_list, 0)

    def _postprocess(self, pred_list):
        line = ''
        for pred in pred_list:
            if pred < 9:
                line += str(pred)
            elif pred < 9 + 10:
                line += str(pred - 9)
            else:
                pred -= 19
                line += chr(ord('A') + pred)
        return line

    def _inference(self, char_img):
        res_list = []
        pred = self.model(char_img)
        for i in range(pred.shape[0]):
            if i == 0:
                _, res = torch.max(pred[i,:9], 0)
                res = res.item()
            elif i == 1:
                _, res = torch.max(pred[i,19:], 0)
                res = res.item() + 19
            else:
                _, res = torch.max(pred[i,9:], 0)
                res = res.item() + 9
            res_list.append(res)
        print("res_list = ", res_list)
        return res_list
