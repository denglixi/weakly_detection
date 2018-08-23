#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: ssd_weakly.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年07月02日 星期一 19时08分23秒
#########################################################################
'''
每一层单独的softmax接在后面
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import collections
import numpy as np
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint

import os

def get_input_fn(filenames,batch_size=1,is_shuffled=False):
    def dataset_input_fn():
        #Get data
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_voc_tf)
        if is_shuffled:
            dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.repeat()
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        #normalize the label for softmax
        labels_sum = tf.reduce_sum(labels,axis=1)
        labels = labels / tf.reshape(labels_sum,(-1,1))
        def preprocess(images):
            images = tf.to_float(images)
            return images
        images = preprocess(images)
        return images,labels
    return dataset_input_fn

def main():
    #get net
    pass
            



if __name__ == '__main__':
  main()

