from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class_n = 45

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
                nn.Linear(256, class_n),
            )

    def forward(self, x):
        x = self.feature(x)
        adaptiveAvgPoolWidth = x.shape[2]
        x = F.avg_pool2d(x, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

class CarIdDataset(Dataset):
    def __init__(self, data_list, mode, weight = 229, height = 229):
        self.data_list = data_list
        self.mode =mode
        self.weight = weight
        self.height = height

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        img_path, label = self.data_list[index]["image_path"], self.data_list[index]["label"]
        img = np.array(Image.open(img_path))
        
        h, w, _ = img.shape
        M = cv2.getAffineTransform(self.data_list[index]["pts"][0], self.data_list[index]["pts"][1])
        img_dst = cv2.warpAffine(img, M, (w, h))
        
        char_img_list = []
        for [x, y] in self.data_list[index]["char_segmentation"]:
            char_img = cv2.resize(img_dst[:, x:y, :], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            char_img = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(char_img)
            char_img_list.append(char_img)
        img = torch.stack(char_img_list,0)
        img = transforms.Compose([])(img)
        y = np.zeros((output_n, class_n))
        for i in range(len(label)):
            y[i, label[i]] = 1
        
        return img, y
    
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = torch.cat([inputs[i] for i in range(inputs.shape[0])], 0)
                labels = torch.cat([labels[i] for i in range(labels.shape[0])], 0)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs).double()
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    _, y = torch.max(labels, 1)
                    running_corrects += torch.sum(preds == y)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_filename = model_file + os.sep + str(epoch) + "checkpoint.pth.tar"
                torch.save({"state_dict":best_model_wts}, model_filename)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            model_filename = model_file + os.sep + "last_checkpoint.pth.tar"
            torch.save({"state_dict":model.state_dict()}, model_filename)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    best_model_filename = model_file + os.sep + "best_checkpoint.pth.tar"
    torch.save({"state_dict":best_model_wts}, best_model_filename)
    final_model_filename = model_file + os.sep + "final_checkpoint.pth.tar"
    torch.save({"state_dict":model.state_dict()}, final_model_filename)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


preprocess_data = "./data.pkl"
model_file = "./model"

pkl_file = open(preprocess_data, 'rb')
data_list = pickle.load(pkl_file)
print(len(data_list))

config_batch_size = 32
train_data_list, val_data_list, _, _ = train_test_split(data_list, data_list, test_size=0.2, random_state=random_seed)
train_gen = CarIdDataset(train_data_list, "train")
train_loader = DataLoader(train_gen,batch_size=config_batch_size,shuffle=True,pin_memory=True,num_workers=2)

val_gen = CarIdDataset(val_data_list, "val")
val_loader = DataLoader(val_gen,batch_size=config_batch_size,shuffle=False,pin_memory=True,num_workers=2)
dataloaders_dict = {"train":train_loader, "val":val_loader}

model_ft = Net()
last_model_filename = model_file + os.sep + "last_checkpoint.pth.tar"
last_model = torch.load(last_model_filename)
model_ft.load_state_dict(last_model["state_dict"])
model_ft.to(device)

params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
criterion = nn.BCEWithLogitsLoss().to(device)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=300)