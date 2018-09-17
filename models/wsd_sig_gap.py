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
from weakly_detection.net_utils.net_utils import pad_to_multiple,conv_hyperarms_fn, filter_features, dict_feature_conv_to_cnls
from weakly_detection.models.SSD_backbone import SSD_Backbone_Net
from weakly_detection.models.cls_relate_feature import cls_relate_feature_conv
from weakly_detection.models.losses import split_layer_softmax 
from weakly_detection.models.metrics import set_metrics
#from tensorflow.contrib.framework import assign_from_checkpoint_fn
from weakly_detection.models.hooks import RestoreHook, get_variable_vgg
from weakly_detection.models.saver import assign_from_checkpoint_fn
slim = tf.contrib.slim

import os
#model fn
def wsd_softmax_gap_model_fn(
        features,
        labels,
        mode,
        params):

    images = features
    feature_maps = SSD_Backbone_Net(images)

    with tf.variable_scope("wsd_feature2cls"):
        #chang feature map to batch * N * N *channel 
        feature_maps = dict_feature_conv_to_cnls(feature_maps,channel_num=20)
        #chang feature map to batch * class_num 
        #cls_relate_features = cls_relate_feature_conv(feature_maps,cls_num=20) # shape of output is same with labels. It is logits
        cls_relate_features = []
        for key,feature in feature_maps.items():
            feature_shape = feature.get_shape()
            feature_shape.assert_has_rank(rank=4)
            cls_relate_features.append(tf.reduce_mean(feature, [1,2]))
    
    probs = []
    preds = []
    for c_feature in cls_relate_features:
        prob = tf.nn.softmax(c_feature)
        probs.append(prob)
        pred = tf.to_int32(prob > params['threshold_T'])
        preds.append(pred)

    if mode != tf.estimator.ModeKeys.PREDICT:
        metrics_ops = set_metrics(preds,labels)
        losses = split_layer_softmax(cls_relate_features,labels)
        total_loss = tf.add_n(losses)
        tf.summary.scalar("total_loss",total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions =  { 
                'image': images,
                }
        for i in range(len(cls_relate_features)):
            predictions["cls_feature_{}".format(i)] = cls_relate_features[i]
        for i in range(len(probs)):
            predictions["prob_{}".format(i)] = probs[i]
        predictions.update(feature_maps)
        return tf.estimator.EstimatorSpec(mode,
                predictions=predictions)

    #accuracy = tf.metrics.accuracy(labels,)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                loss=total_loss,
                eval_metric_ops=metrics_ops)

    #if mode == tf.estimator.ModeKeys.TRAIN:
    assert mode == tf.estimator.ModeKeys.TRAIN
    training_hooks = []

    #resotre from vgg
    if params!=None and 'restore_from_pretrain_weight' in params and params['restore_from_pretrain_weight']:
        weight_path = params['pretrain_weight_path']
        var_list = tf.global_variables()
        init_fn = assign_from_checkpoint_fn(weight_path,var_list,ignore_missing_vars=True)
        training_hooks.append(RestoreHook(init_fn))

    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
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

