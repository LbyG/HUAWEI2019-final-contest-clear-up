from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_validation import train_test_split
from pretrainedmodels.models import bninception
from imgaug import augmenters as iaa

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time

import matplotlib.pyplot as plt
from PIL import Image
import pickle

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import copy

from preprocess import *

img_w, img_h = 64, 64
random_seed = 4050
config_batch_size = 4
class_n = (9 + 10 + 26)
output_n = 9
num_epochs = 100
feature_extract = True
use_pretrained=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

char_to_index = {"深":0, "秦":1, "京":2, "海":3, "成":4, "南":5, "杭":6, "苏":7, "松":8}
print(char_to_index)

label_file = "./data/train-data-label.txt"
image_file = "./data/data"
model_file = "./model"
preprocess_data = "./data.pkl"
data_list = []
with open(label_file, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline().strip() # 整行读取数据
        if not lines:
            break
        lines = lines.split(",  ")
        image_path = os.path.join(image_file, lines[1])
        label = [];
        label.append(char_to_index[lines[0][0]])
        for i in range(1, len(lines[0])):
            if '0' <= lines[0][i] and lines[0][i] <= '9':
                label.append(9 + ord(lines[0][i]) - ord('0'))
            else:
                label.append(9 + 10 + ord(lines[0][i]) - ord('A'))
        data_list.append({"image_path": image_path, "label":label})

for count in range(len(data_list)):
    data = data_list[count]
    print("count = ", count, "data = ", data)
    pts1, pts2, char_segmentation = preprocess(data["image_path"])
    data["pts"] = [pts1, pts2]
    data["char_segmentation"] = char_segmentation
    
    data_pkl_path = data["image_path"].replace("jpg", "pkl").replace("data", "preprocess")
    output = open(data_pkl_path, 'wb')
    pickle.dump(data, output)
    output.close()
    
preprocess_data_list = []
for data in data_list:
    data_pkl_path = data["image_path"].replace("jpg", "pkl").replace("data", "preprocess")
    if os.path.exists(data_pkl_path):
        with open(data_pkl_path, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file)
            preprocess_data_list.append(pkl_data)
print(len(preprocess_data_list))
output = open(preprocess_data, 'wb')
pickle.dump(preprocess_data_list, output)
output.close()
