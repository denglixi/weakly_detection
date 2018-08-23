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
from weakly_detection.slim.nets import vgg_atrous_ssd
from weakly_detection.net_utils.feature_generator import multi_resolution_feature_maps
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint
from weakly_detection.net_utils.net_utils import pad_to_multiple,conv_hyperarms_fn, filter_features, dict_feature_conv_to_cls
from weakly_detection.models.SSD_backbone import SSD_Backbone_Net
from weakly_detection.models.cls_relate_feature import cls_relate_feature_conv
from weakly_detection.models.losses import split_layer_softmax 

from tensorflow.contrib.framework import assign_from_checkpoint_fn
from weakly_detection.models.hooks import RestoreHook, get_variable_vgg
slim = tf.contrib.slim


import os

#model fn
def wsd_softmax_conv_model_fn(
        features,
        labels,
        mode,
        params):

    feature_maps = SSD_Backbone_Net(features)

    #chang feature map to batch * N * N *channel 
    feature_maps = dict_feature_conv_to_cls(feature_maps,channel_num=256)
    #chang feature map to batch * class_num 
    cls_relate_features = cls_relate_feature_conv(feature_maps,cls_num=20) # shape of output is same with labels. It is logits
    
    probs = []
    preds = []
    for c_feature in cls_relate_features:
        prob = tf.nn.softmax(c_feature)
        probs.append(prob)
        pred = tf.to_int32(prob > params['threshold_T'])
        preds.append(pred)

    metrics_ops = {}
    for i,pred in enumerate(preds):
        recall = tf.metrics.recall(labels,pred)
        metrics_ops["recall_{}".format(i)] = recall
        tf.summary.scalar("recall_layer_{}".format(i),recall[0])

        accu = tf.metrics.accuracy(labels,pred)
        metrics_ops["accu_{}".format(i)] = accu
        tf.summary.scalar("accu_{}".format(i),accu[0])

        precision  = tf.metrics.precision(labels,pred)
        metrics_ops["precision_{}".format(i)] = precision
        tf.summary.scalar("precision_{}".format(i),precision[0])

    losses = split_layer_softmax(cls_relate_features,labels)
    total_loss = tf.add_n(losses)
    tf.summary.scalar("total_loss",total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    #accuracy = tf.metrics.accuracy(labels,)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                loss=total_loss,
                eval_metric_ops=metrics_ops)
        pass

    #if mode == tf.estimator.ModeKeys.TRAIN:
    assert mode == tf.estimator.ModeKeys.TRAIN

    training_hooks = []

    #resotre from vgg
    if params!=None and 'restore_from_vgg' in params and params['restore_from_vgg']:
        vgg_path = params['vgg_path']
        var_list = tf.global_variables()#get_variable_vgg(vgg_path)
        init_fn = assign_from_checkpoint_fn(vgg_path,var_list,ignore_missing_vars=True)
        training_hooks.append(RestoreHook(init_fn))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,
            loss=total_loss, 
            train_op=train_op,
            eval_metric_ops=metrics_ops,
            training_hooks=training_hooks)


def main():
    #get net
    pass 



if __name__ == '__main__':
  main()

