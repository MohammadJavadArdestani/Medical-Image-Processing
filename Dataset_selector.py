import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import time
import copy
from random import shuffle

import tqdm.notebook as tqdm

import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report
from PIL import Image
import cv2

import os
import shutil

# df = pd.read_csv('./covid-chestxray-dataset/metadata.csv')
# selected_df = df[df.finding=="Pneumonia/Viral/COVID-19"]
# selected_df = selected_df[(selected_df.view == "AP") | (selected_df.view == "PA")]
# selected_df.head(2)

# images = selected_df.filename.values.tolist()

# os.makedirs('./COVID19-DATASET/train/covid19')
# os.makedirs('./COVID19-DATASET/train/normal')

# COVID_PATH = './COVID19-DATASET/train/covid19'
# NORMAL_PATH = './COVID19-DATASET/train/normal'
# DATA_PATH = './COVID19-DATASET/train'

# for image in os.listdir('./covid-chestxray-dataset/images')[100:550]:
#     shutil.copy(os.path.join('./covid-chestxray-dataset/images', image), os.path.join(COVID_PATH, image))


# for image in os.listdir('./ChestXRay2017/chest_xray/train/NORMAL')[100:300]:
#     shutil.copy(os.path.join('./ChestXRay2017/chest_xray/train/NORMAL', image), os.path.join(NORMAL_PATH, image))

# for image in os.listdir('./COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Normal/images')[100:300]:
#     shutil.copy(os.path.join('./COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Normal/images', image), os.path.join(NORMAL_PATH, image))

os.makedirs('./COVID19-DATASET/test/covid19')
os.makedirs('./COVID19-DATASET/test/normal')

COVID_TEST1 = './COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/COVID/images'
COVID_TEST2 = './covid-chestxray-dataset/images'
NORMAL_TEST1 = './ChestXRay2017/chest_xray/train/NORMAL'
NORMAL_TEST2 = './COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/Normal/images'
for image in os.listdir(COVID_TEST1)[:100]:
    shutil.copy(os.path.join(COVID_TEST1, image), os.path.join('./COVID19-DATASET/test/covid19', image))
for image in os.listdir(COVID_TEST2)[:100]:
    shutil.copy(os.path.join(COVID_TEST2, image), os.path.join('./COVID19-DATASET/test/covid19', image))
for image in os.listdir(NORMAL_TEST1)[500:600]:
    shutil.copy(os.path.join(NORMAL_TEST1, image), os.path.join('./COVID19-DATASET/test/normal', image))
for image in os.listdir(NORMAL_TEST2)[:100]:
    shutil.copy(os.path.join(NORMAL_TEST2, image), os.path.join('./COVID19-DATASET/test/normal', image))

