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
from weakly_detection.slim.nets import vgg_atrous_ssd
from weakly_detection.net_utils.feature_generator import multi_resolution_feature_maps
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint
from weakly_detection.net_utils.net_utils import pad_to_multiple, conv_hyperarms_fn, filter_features, get_class_relate_feature

slim = tf.contrib.slim

#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()



#net, end_points = vgg_atrous_ssd.vgg_16(
#    inputs,
#    num_classes=21,
#    is_training=True,
#    dropout_keep_prob=0.5,
#    spatial_squeeze=True,
#    scope='vgg_16',
#    fc_conv_padding='VALID',
#    global_pool=False)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def SSD_Backbone_Net(images):
    #ssd backbone net based on vgg16
    feature_map_layout = {
        'from_layer': ['conv4_3', 'fc7', '', '',
                       '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 256],
        'layer_stride': [-1,-1,2,2,1,1],
        'layer_padding': ['','', 'SAME','SAME','VALID','VALID'],
        'use_explicit_padding': False,
        'use_depthwise': False,
    }

    with slim.arg_scope(conv_hyperarms_fn()):
        with tf.variable_scope('vgg_16', reuse=None) as scope:
            _, image_features = vgg_atrous_ssd.vgg_16(
                pad_to_multiple(images, multiple=1),
                scope=scope)
            image_features = filter_features(image_features)
            feature_maps = multi_resolution_feature_maps(
                feature_map_layout=feature_map_layout,
                insert_1x1_conv=True,
                image_features=image_features)

    return feature_maps
