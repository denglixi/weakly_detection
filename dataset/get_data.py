#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: dataset/get_data.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月23日 星期四 09时32分15秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from weakly_detection.data_decoder import _parse_voc_tf

def get_data(filenames,batch_size=1,is_shuffled=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_voc_tf)
    if is_shuffled:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.repeat()
    batched_dataset = dataset.batch(batch_size)

    iterator = batched_dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()
    return images, labels

def main():
  pass

if __name__ == '__main__':
  main()

