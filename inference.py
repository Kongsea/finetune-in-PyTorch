# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-08-03 16:13:30
"""
from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm

from create_dataset_from_csv import Create_Dataset_From_CSV

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = Create_Dataset_From_CSV('valid.csv', transform=data_transforms,
                                         target_size=224, retrieve_paths=True)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, num_workers=4)

use_gpu = torch.cuda.is_available()


def check_test(model, dataloaders, use_gpu):
  model.eval()
  apredictions = []
  ascores = []
  apaths = []
  for inputs, _, paths in tqdm(dataloaders):
    if use_gpu:
      inputs = inputs.cuda()
    with torch.no_grad():
      inputs = Variable(inputs)
      outputs = model(inputs)
      softmax = nn.Softmax(dim=1)(outputs)
    scores = np.max(softmax.data.cpu().numpy(), 1)
    _, preds = torch.max(outputs, 1)
    preds = preds.data.cpu().numpy()
    apredictions.extend(preds)
    ascores.extend(scores)
    apaths.extend(paths)

  return apredictions, ascores, apaths


def move_test(path, label_map, predictions, scores, paths):
  for _pred, _score, _path in zip(predictions, scores, paths):
    pp = os.path.join(path, label_map[_pred])
    if not os.path.exists(pp):
      os.makedirs(pp)
    shutil.copy(
        _path, '{}/{:.3f}_{}.jpg'.format(pp, _score, _path.rpartition('/')[-1][:-4]))


def test_model(model, dataloaders, label_map):

  predictions, scores, paths = check_test(model, dataloaders, use_gpu)
  move_test('test_results',
            label_map, predictions, scores, paths)


def main():

  with open('classes.txt') as f:
    classes = [c.strip() for c in f.readlines()]
  label_map = {i: v for i, v in enumerate(classes)}

  model_ft = models.resnet101()
  num_ftrs = model_ft.fc.in_features

  model_ft.fc = nn.Sequential(
      nn.Linear(num_ftrs, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.4),
      nn.Linear(256, len(classes)),
  )

  criterion = nn.CrossEntropyLoss()

  if use_gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    criterion = criterion.cuda()
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))

  model_ft.load_state_dict(torch.load('checkpoint/resnet101_0.974.t7')['state_dict'])
  print('Model loaded.')

  test_model(model_ft, dataloaders, label_map)


if __name__ == '__main__':
  main()
