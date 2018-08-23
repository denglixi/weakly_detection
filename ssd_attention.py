#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: ssd_weakly.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年07月02日 星期一 19时08分23秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import collections
import numpy as np
from weakly_detection.dataset.pascal_voc import Get_input_fn
from weakly_detection.models.model_factory import get_model

slim = tf.contrib.slim

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

tf.logging.set_verbosity(tf.logging.INFO)

def main():
    #get net

    input_fn = Get_input_fn(16)
    MODEL_NAME = 'softmax_split'
    model_fn = get_model(MODEL_NAME)
    classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir='./results/{}/'.format(MODEL_NAME),
            )
    classifier.train(input_fn=input_fn)
            



if __name__ == '__main__':
  main()

