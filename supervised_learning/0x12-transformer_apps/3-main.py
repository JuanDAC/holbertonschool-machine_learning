#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('2-dataset').Dataset

data = Dataset()
print('got here')
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
