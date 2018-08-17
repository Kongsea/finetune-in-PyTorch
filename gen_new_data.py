#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os

path = 'data'


def gen_lines(path):
  lines = []
  for f in os.listdir(path):
    if f.endswith('.jpg') and not f.startswith('.'):
      if os.stat(os.path.join(path, f)).st_size != 0:
        lines.append(os.path.join(path, f))
  return lines


all_lines = []
for dirpath, dirnames, filenames in os.walk(path):
  lines = gen_lines(dirpath)
  lines = [[line, '其他'] for line in lines]
  all_lines.extend(lines)

print(len(all_lines))

with open('new_data.csv', 'wb') as f:
  cw = csv.writer(f)
  for line in all_lines:
    cw.writerow(line)
