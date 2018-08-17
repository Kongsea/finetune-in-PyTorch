# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-07-20 20:32:34
"""
from __future__ import absolute_import, division, print_function

import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


def init_params(net):
  '''Init layer parameters.'''
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      init.kaiming_normal(m.weight, mode='fan_out')
      if m.bias:
        init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant(m.weight, 1)
      init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.normal(m.weight, std=1e-3)
      if m.bias:
        init.constant(m.bias, 0)


def cal_print_cm(labels, predictions, label_map, print_=True):
  cm = confusion_matrix(labels, predictions)
  precision = {}
  recall = {}
  pr = 0
  ap = ar = 0
  for i, c in enumerate(cm):
    pr += c[i]
    ap += sum(cm[:, i])
    ar += sum(c)
    precision[label_map[i]] = c[i]/sum(cm[:, i])
    recall[label_map[i]] = c[i]/sum(c)

  precision['all'] = pr/ap
  recall['all'] = pr/ar

  if print_:
    for i, c in enumerate(cm):
      print('{} {}'.format(label_map[i], c))

  return precision, recall


def check_training(model, dataloaders, label_map, use_gpu, print_=True):
  model.eval()
  apredictions = []
  alabels = []
  for inputs, labels, _ in dataloaders:
    if use_gpu:
      inputs = inputs.cuda()
    with torch.no_grad():
      inputs = Variable(inputs)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    apredictions.extend(preds.data.cpu().numpy())
    alabels.extend(labels.data.numpy())

  precision, recall = cal_print_cm(alabels, apredictions, label_map, print_=print_)

  return precision, recall


def check(model, dataloaders, label_map, use_gpu, print_=True):
  model.eval()
  error_cls = {}
  apredictions = []
  alabels = []
  ascores = []
  for inputs, labels, paths in dataloaders:
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
    labels = labels.data.numpy()
    alabels.extend(labels)
    ascores.extend(scores)
    for l, p, f in zip(labels, preds, paths):
      if p != l:
        error_cls.setdefault(label_map[l], []).append((f, label_map[p]))

  precision, recall = cal_print_cm(alabels, apredictions, label_map, print_=print_)

  return precision, recall, error_cls, ascores


def print_precision_recall(label_map, precision, recall, phase='train'):
  lm = sorted(label_map.values())
  lm.append('all')
  print('=' * 10 + 'precision of {}'.format(phase) + '=' * 10)
  for k in lm:
    if k not in precision:
      print('{} -> {:.3f}'.format(k, 0))
      continue
    print('{} -> {:.3f}'.format(k, precision[k]))
  print('=' * 10 + 'recall of {}'.format(phase) + '=' * 10)
  for k in lm:
    if k not in recall:
      print('{} -> {:.3f}'.format(k, 0))
      continue
    print('{} -> {:.3f}'.format(k, recall[k]))


def move_error(path, label_map, precision, recall, error_cls, phase='train'):
  lm = sorted(label_map.values())
  lm.append('all')
  if not os.path.isdir(path):
    os.makedirs(path)
  print_precision_recall(label_map, precision, recall, phase=phase)
  for k in lm:
    if k not in error_cls:
      continue
    print('{} error in {}'.format(phase, k))
    for f, p in error_cls[k]:
      print('{} -> {}'.format(f, p))
      shutil.copy(
          f, '{}/{}_{}_{}_{}.jpg'.format(path, phase, f.rpartition('/')[-1][:-4], k, p))
