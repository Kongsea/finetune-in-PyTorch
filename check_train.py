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

SCORE_THRESHOLD = 0.9

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = Create_Dataset_From_CSV('train.csv', transform=data_transforms,
                                         target_size=224, retrieve_paths=True)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=128, num_workers=4)

use_gpu = torch.cuda.is_available()


def check_train(model, dataloaders, use_gpu):
  model.eval()
  apredictions = []
  alabels = []
  ascores = []
  apaths = []
  for inputs, labels, paths in tqdm(dataloaders):
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
    alabels.extend(labels.numpy().tolist())
    ascores.extend(scores)
    apaths.extend(paths)

  return apredictions, alabels, ascores, apaths


def move_test(path, label_map, predictions, labels, scores, paths):
  for _pred, _label, _score, _path in zip(predictions, labels, scores, paths):
    if _score > SCORE_THRESHOLD and _pred == _label:
      continue
    pp = os.path.join(path, label_map[_label])
    if not os.path.exists(pp):
      os.makedirs(pp)
    shutil.move(
        _path, '{}/{:.3f}_{}_{}.jpg'.format(pp, _score, _path.rpartition('/')[-1][:-4], label_map[_pred]))


def test_model(model, dataloaders, label_map):

  predictions, labels, scores, paths = check_train(model, dataloaders, use_gpu)
  move_test('/home/konghaiyang/kong/scene_classifier/data/data0810_train',
            label_map, predictions, labels, scores, paths)


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

  if use_gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))

  model_ft.load_state_dict(torch.load('checkpoint/resnet101_exp_0815_0.974.t7')['state_dict'])

  test_model(model_ft, dataloaders, label_map)


if __name__ == '__main__':
  main()
