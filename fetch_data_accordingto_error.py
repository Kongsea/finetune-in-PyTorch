# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-08-03 15:02:22
"""
from __future__ import absolute_import, division, print_function

import os
import shutil
from tqdm import tqdm

error_path = '/home/fallingdust/4T/kong/pytorch/pytorch-cifar/error'

source_path = '/home/konghaiyang/kong/scene_classifier/data/data0810'

target_path = os.path.join(source_path, 'fetch')

error_files = [f for f in os.listdir(error_path)]

print(len(error_files))


def isfloat(value):
  try:
    value = float(value)
    if value <= 1:
      return True
    else:
      return False
  except ValueError:
    return False


for f in tqdm(error_files):
  if f.startswith('train') or f.startswith('valid'):
    bn = f[6:-4]
  elif f.startswith('val'):
    bn = f[4:-4]
  else:
    bn = f[:-4]
  parts = bn.split('_')
  alt_name = None
  alt_path = None
  if isfloat(parts[0]):
    ori_name = parts[1:-2]
    alt_name = parts[:-2]
  else:
    ori_name = parts[:-2]
  ori_class = parts[-2]
  cal_class = parts[-1]
  ori_name = '_'.join(ori_name)
  if alt_name:
    alt_name = '_'.join(alt_name)
    alt_path = os.path.join(source_path, ori_class, '{}.jpg'.format(alt_name))
  ori_path = os.path.join(source_path, ori_class, '{}.jpg'.format(ori_name))
  if not os.path.exists(os.path.join(target_path, ori_class)):
    os.makedirs(os.path.join(target_path, ori_class))
  if os.path.exists(ori_path):
    os.remove(os.path.join(error_path, f))
    shutil.move(ori_path, os.path.join(target_path, ori_class, '{}.jpg'.format(ori_name)))
  elif alt_path is not None and os.path.exists(alt_path):
    os.remove(os.path.join(error_path, f))
    shutil.move(alt_path, os.path.join(target_path, ori_class, '{}.jpg'.format(ori_name)))
  else:
    print(os.path.join(error_path, f))
