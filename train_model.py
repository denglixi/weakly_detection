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
from weakly_detection.dataset.pascal_voc import get_input_fn
from weakly_detection.models.model_factory import get_model_fn

slim = tf.contrib.slim

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.logging.set_verbosity(tf.logging.INFO)

def main():
    #get net
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000, 
                                        keep_checkpoint_max=0)

    train_input_fn = get_input_fn("./pascal_train.record",16)
    eval_input_fn = get_input_fn("./pascal_val.record",16)
    MODEL_NAME = 'softmax_gap'
    #MODEL_NAME = 'softmax_split'
    model_fn = get_model_fn(MODEL_NAME)
    training_params = {}
    training_params["restore_from_pretrain_weight"] = True
    training_params["pretrain_weight_path"] = '../../Experiments/detection/models/research/Experiments/weights/vgg_16.ckpt'
    #training_params["pretrain_weight_path"] = "./results/softmax_conv_cls_feature/model.ckpt-275000"
    training_params["threshold_T"] = 0.1
    training_params["lr"] = 0.001

    eval_params = {}   
    eval_params["threshold_T"] = 0.1
    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir='./results/estimator_{}/'.format(MODEL_NAME),
            params=training_params,
            config=run_config
            )
    
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)

    eval_spec = tf.estimator.EvalSpec(
                name="eval",
                input_fn=eval_input_fn,
                start_delay_secs=120,
                throttle_secs=600)
    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
            



if __name__ == '__main__':
  main()

