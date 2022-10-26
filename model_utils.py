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

import matplotlib.pyplot as plt
import numpy as np
import itertools


class_names = ['covid19', 'normal']

def train_validate_model(model, criterion, optimizer, scheduler, dataloaders, data_sizes, device, num_epochs=10):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            current_loss = 0.0
            current_corrects = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase, leave=False):

                optimizer.zero_grad()

                # Load data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward Phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Change Learning Rate for the next Epoch        
                if phase == 'train':
                    scheduler.step()

                # Update Loss and correct predictions per batch
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            # Update Loss and correct predictions per eppoch
            epoch_loss = current_loss / data_sizes[phase]
            epoch_acc = current_corrects.double() / data_sizes[phase]

            # Loss and accuracy Report
            if phase == 'val':
                print('{} Loss: {:.4f} | {} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, phase, epoch_acc))
            else:
                print('{} Loss: {:.4f} | {} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, phase, epoch_acc))

            # Saving best Model before Early stoping
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # load the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model

def visualize_test(model, num_images, dataloaders, device):
    was_training = model.training
    model.eval()
    images_handeled = 0
    ax = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            for j in range(inputs.size()[0]):
                print('Actual: {} predicted: {}'.format(class_names[labels[j].item()],class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j], (5,5))
