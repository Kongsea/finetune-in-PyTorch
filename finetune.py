# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   @Author: Kong Haiyang
   @Date: 2018-07-20 15:41:38
"""
from __future__ import absolute_import, division, print_function

import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models, transforms

from create_dataset_from_csv import Create_Dataset_From_CSV
from utils import check, check_training, move_error, print_precision_recall

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

image_datasets = {'train': Create_Dataset_From_CSV('train.csv', train=True, fliplr=True, rotate=True,
                                                   color=True, cutout=True, crop=False, augment=True,
                                                   transform=data_transforms['train'],
                                                   target_size=224, retrieve_paths=True),
                  'valid': Create_Dataset_From_CSV('valid.csv', transform=data_transforms['valid'],
                                                   target_size=224, retrieve_paths=True)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=12)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_acc = 0.0
  print_step = 3

  classes2idx = image_datasets['train'].classes2idx
  label_map = {v: k for k, v in classes2idx.iteritems()}

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 20)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
      if phase == 'train':
        scheduler.step()
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels, _ in dataloaders[phase]:

        if use_gpu:
          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
          inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{}: Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))

      if epoch % print_step == 0:
        check_training(model, dataloaders[phase], label_map, use_gpu)

      if phase == 'valid' and epoch_acc > best_acc:

        old_model = './checkpoint/resnet101_exp_{:.3f}.t7'.format(best_acc)
        if os.path.exists(old_model):
          os.remove(old_model)

        best_acc = epoch_acc

        print('Saving model...')
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        torch.save(state, './checkpoint/resnet101_exp_{:.3f}.t7'.format(best_acc))
        print('Saving model completed..')

      torch.cuda.empty_cache()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  precision, recall, error_cls, _ = check(model, dataloaders['train'], label_map, use_gpu)
  print_precision_recall(label_map, precision, recall, 'train')
  move_error('error', label_map, precision, recall, error_cls, 'train')
  precision, recall, error_cls, _ = check(model, dataloaders['valid'], label_map, use_gpu)
  print_precision_recall(label_map, precision, recall, 'valid')
  move_error('error', label_map, precision, recall, error_cls, 'valid')


def main():
  model_ft = models.resnet101(pretrained=True)
  num_ftrs = model_ft.fc.in_features

  model_ft.fc = nn.Sequential(
      nn.Linear(num_ftrs, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.4),
      nn.Linear(256, len(class_names)),
  )

  for param in model_ft.parameters():
    param.requires_grad = False
  for param in model_ft.layer3.parameters():
    param.requires_grad = True
  for param in model_ft.layer4.parameters():
    param.requires_grad = True
  for param in model_ft.fc.parameters():
    param.requires_grad = True

  criterion = nn.CrossEntropyLoss()

  if use_gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    criterion = criterion.cuda()
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))

  optimizer_ft = optim.SGD([p for p in model_ft.parameters() if p.requires_grad],
                           lr=0.01, momentum=0.9, weight_decay=5e-5)

  # Decay LR by a factor of 0.25 every 10 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.25)

  if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

  train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)


if __name__ == '__main__':
  main()
