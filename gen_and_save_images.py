#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models, transforms

from create_dataset_from_csv import Create_Dataset_From_CSV

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_set = Create_Dataset_From_CSV('train.csv', fliplr=True, rotate=True,
                                    color=True, cutout=True, crop=False, augment=True,
                                    transform=data_transforms['train'],
                                    target_size=224, retrieve_paths=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                           shuffle=True, num_workers=4)

train_size = len(train_set)

dset_classes = train_set.classes

for batch_idx, (images, labels, paths) in enumerate(train_loader):
  images = images.numpy().astype(np.uint8)
  print(images.shape)
  images = np.transpose(images, [0, 2, 3, 1])
  print(images.shape)
  labels = labels.numpy()
  print(labels.shape)
  num = labels.shape[0]
  for i in range(num):
    cv2.imwrite('fetch/{}_{}_{}.jpg'.format(batch_idx, i, labels[i]),
                (images[i]+1)/2*255)
