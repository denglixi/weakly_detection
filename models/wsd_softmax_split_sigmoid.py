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
import os
from weakly_detection.slim.nets import vgg_atrous_ssd
from weakly_detection.net_utils.feature_generator import multi_resolution_feature_maps
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint
from weakly_detection.net_utils.net_utils import pad_to_multiple, conv_hyperarms_fn, filter_features, get_class_relate_feature
from weakly_detection.models.SSD_backbone import SSD_Backbone_Net

slim = tf.contrib.slim

#model fn
def wsd_softmax_split_sigmoid_model(
        features,
        labels,
        mode,
        params):

    feature_maps = SSD_Backbone_Net(features)

    #chang feature map to N * N * class
    feature_maps = get_class_relate_feature(feature_maps)

    class_relate_features = []
    for key,feature in feature_maps.items():
        feature_shape = feature.get_shape()
        feature_shape.assert_has_rank(rank=4)
        class_relate_features.append(tf.reduce_sum(tf.sigmoid(feature), [1,2]))
        
    #logits = tf.add_n(class_relate_features)
    losses = []
    for i,c_feature in enumerate(class_relate_features):
        prob = tf.nn.softmax(c_feature)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_float(labels), logits=c_feature, name='loss'))
        tf.summary.scalar("loss_{}".format(i),loss)
        losses.append(loss)

    total_loss = tf.add_n(losses)
    tf.summary.scalar("total_loss",total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    #accuracy = tf.metrics.accuracy(labels,)
    if mode == tf.estimator.ModeKeys.EVAL:
        pass

    #if mode == tf.estimator.ModeKeys.TRAIN:
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,loss=loss, train_op=train_op)#, training_hooks=training_hooks)


def main():
    #get net
    pass 



if __name__ == '__main__':
  main()

