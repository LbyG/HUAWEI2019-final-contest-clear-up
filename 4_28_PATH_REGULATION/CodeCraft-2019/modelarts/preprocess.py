import numpy as np
import os
import cv2
import random
import math

import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import scipy.signal as signal

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

def preprocess(img_path):
    diff_threshold = 80
    search_step = 10
    point_width = 2
    flag_dst_threshold = 15
    
    img = Image.open(img_path)
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
    """
    count = np.zeros((h, w))
    max_count = 0
    max_i, max_j = 0, 0
    for i in range(0, h, h // search_step):
        base_x = i#(random.randint(0, h // search_step) + i) % h
        for j in range(0, w, w // search_step):
            base_y = j#(random.randint(0, w // search_step) + j) % w
            for x in range(h):
                for y in range(w):
                    if diff(img[base_x, base_y], img[x, y]) < diff_threshold:
                        count[base_x, base_y] += 1
            if max_count < count[base_x, base_y]:
                max_count = count[base_x, base_y]
                max_i = base_x
                max_j = base_y
    flag = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if diff(img[i, j], img[max_i, max_j]) < diff_threshold:
                flag[i, j] = 255
    """
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
    
    