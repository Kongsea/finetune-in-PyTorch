#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class Create_Dataset_From_CSV(Dataset):

  def __init__(self, csv_file, train=False, fliplr=False, rotate=False,
               color=False, cutout=False, crop=False, augment=False,
               transform=None, target_transform=None,
               target_size=224, retrieve_paths=False):
    self.csv_file = csv_file
    self.train = train
    self.fliplr = fliplr
    self.rotate = rotate
    self.color = color
    self.cutout = cutout
    self.crop = crop
    self.augment = augment
    self.transform = transform
    self.target_transform = target_transform
    self.target_size = target_size
    self.retrieve_paths = retrieve_paths

    self.classes = []
    self.classes2idx = {}
    self.images = []
    self.labels = []

    self.get_csv_lines_and_classes()
    self.make_dataset()

  def __getitem__(self, index):
    path = self.images[index]
    label = self.labels[index]
    image = self.image_loader(path)

    if self.transform is not None:
      image = self.transform(image)

    if self.target_transform is not None:
      label = self.target_transform(label)

    if self.retrieve_paths:
      return image, label, path
    else:
      return image, label

  def __len__(self):
    return len(self.images)

  def image_loader(self, path):
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    large = np.min(img.shape[:-1]) > 800
    if large:
      img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    if self.fliplr:
      seq = iaa.Fliplr(p=0.3)
      img = seq.augment_image(img)

    if self.rotate and np.random.random() < 0.3:
      seq = iaa.Affine(rotate=(-10, 10))
      img = seq.augment_image(img)

    if self.color and np.random.random() < 0.5:
      seq = iaa.SomeOf((2, 4), [
          iaa.AdditiveGaussianNoise(loc=(0.8, 1.2), scale=(0, 3)),
          iaa.Add((-10, 10), per_channel=0.5),
          iaa.Multiply((0.9, 1.1), per_channel=0.5),
          iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5)
      ])
      img = seq.augment_image(img)

    if self.cutout:
      seq = iaa.CoarseDropout(p=0.05, size_px=(3, 5))
      img = seq.augment_image(img)

    if self.crop and np.random.random() < 0.6 and large:
      seq = iaa.Sequential([
          iaa.Crop(percent=((0.05, 0.1), (0.05, 0.1), (0.05, 0.1), (0.05, 0.1)), keep_size=False)
      ])
      img = seq.augment_image(img)

    image = self.pad_image(img, self.target_size)

    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    return image

  def pad_image(self, img, size=224):
    image = np.zeros((size, size, 3), dtype=np.float)
    image[:, :, :] = [104, 117, 124]
    height = img.shape[0]
    width = img.shape[1]
    if max(height, width) > size:
      if height > width:
        ratio = size / height
        width = int(ratio * width)
        img = cv2.resize(img, (width, size))
      else:
        ratio = size / width
        height = int(ratio * height)
        img = cv2.resize(img, (size, height))
    height = img.shape[0]
    width = img.shape[1]
    if height >= width:
      ratio = size / height
      width = int(ratio * width)
      img = cv2.resize(img, (width, size))
      image[:, int((size - width) / 2):int((size - width) / 2) + width, :] = img
    else:
      ratio = size / width
      height = int(ratio * height)
      img = cv2.resize(img, (size, height))
      image[int((size - height) / 2):int((size - height) / 2) + height, :, :] = img
    return image

  def get_csv_lines_and_classes(self):
    with open(self.csv_file) as f:
      _lines = [line.strip() for line in f.readlines()]

    lines = []
    for line in _lines:
      path = line.split(',')[0]
      if os.path.exists(path):
        lines.append(line)

    self.csv_lines = []

    all_lines = {}
    for line in lines:
      cls = line.split(',')[1]
      all_lines.setdefault(cls, []).append(line)

    class_count = {}
    for cls in all_lines:
      class_count[cls] = len(all_lines[cls])
    self.classes = sorted(class_count, key=class_count.get, reverse=True)
    with open('classes.txt') as f:
      self.classes = [c.strip() for c in f.readlines()]
    self.classes2idx = {cls: i for i, cls in enumerate(self.classes)}
    if self.train:
      with open('classes.txt', 'w') as f, open('class_count.txt', 'w') as fc:
        for c in self.classes:
          f.write('{}\n'.format(c))
          fc.write('{} {}\n'.format(c, class_count[c]))

    if self.augment:
      for cls in all_lines:
        aug = 1 if len(all_lines[cls]) > 100 else 2
        self.csv_lines.extend(all_lines[cls] * aug)
    else:
      for cls in all_lines:
        self.csv_lines.extend(all_lines[cls])

    np.random.shuffle(self.csv_lines)

  def make_dataset(self):
    for line in self.csv_lines:
      filename, target = line.split(',')

      if self._is_image_file(filename):
        self.images.append(filename)
        self.labels.append(self.classes2idx[target])

  def _is_image_file(self, filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
