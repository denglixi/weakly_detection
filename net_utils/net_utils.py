#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: net_utils.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月16日 星期四 14时59分39秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import collections
import numpy as np
from weakly_detection.slim.nets import vgg_atrous_ssd
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint

slim = tf.contrib.slim

def main():
  pass

def pad_to_multiple(tensor, multiple):
    tensor_shape = tensor.get_shape()
    tensor_shape.assert_has_rank(rank=4)
    batch_size = tensor_shape[0].value
    tensor_height = tensor_shape[1].value
    tensor_width = tensor_shape[2].value
    tensor_depth = tensor_shape[3].value

    if batch_size is None:
        batch_size = tf.shape(tensor)[0]

    if tensor_height is None:
        tensor_height = tf.shape(tensor)[1]
        padded_tensor_height = tf.to_int32(
            tf.ceil(tf.to_float(tensor_height) / tf.to_float(multiple))) * multiple
    else:
        padded_tensor_height = int(math.ceil(float(tensor_height) / multiple) * multiple)

    if tensor_width is None:
        tensor_width = tf.shape(tensor)[2]
        padded_tensor_width = tf.to_int32(
            tf.ceil(tf.to_float(tensor_width) / tf.to_float(multiple))) * multiple
    else:
        padded_tensor_width = int(
            math.ceil(float(tensor_width) / multiple) * multiple)

    if tensor_depth is None:
        tensor_depth = tf.shape(tensor)[3]

    # Use tf.concat instead of tf.pad to preserve static shape
    if padded_tensor_height != tensor_height:
        height_pad = tf.zeros([
            batch_size, padded_tensor_height - tensor_height, tensor_width,
            tensor_depth
        ])
        tensor = tf.concat([tensor, height_pad], 1)
    if padded_tensor_width != tensor_width:
        width_pad = tf.zeros([
            batch_size, padded_tensor_height, padded_tensor_width - tensor_width,
            tensor_depth
        ])
        tensor = tf.concat([tensor, width_pad], 2)

    return tensor

def conv_hyperarms_fn():
  affect_ops = [slim.conv2d, slim.conv2d_transpose]
  with slim.arg_scope(affect_ops,
                      weights_initializer=tf.initializers.truncated_normal(mean=0.0,stddev=0.03),
                      weights_regularizer=slim.l2_regularizer(scale=float(0.004)),
                      activation_fn=tf.nn.relu6,
                      normalizer_fn=slim.batch_norm) as sc:
      return sc

def filter_features(image_features):
    filtered_image_feature = dict({})
    for key,feature in image_features.items():
        feature_name = key.split('/')[-1]
        if feature_name in ['conv4_3','fc7']:
            filtered_image_feature[feature_name] = feature
    return filtered_image_feature

def get_class_relate_feature(feature_maps, class_num=20):
    with tf.name_scope("class_relate_feature"):
        for key, feature in feature_maps.items():
            feature_maps[key] = tf.layers.conv2d(feature, 
                    filters=class_num, 
                    kernel_size=(3,3),
                    padding='same')
    return feature_maps

def dict_feature_conv_to_cls(feature_maps, channel_num=20):
    '''
    use 3x3 conv the features in dict to [b, n,n,cls_num]
    '''
    with tf.name_scope("class_relate_feature"):
        for key, feature in feature_maps.items():
            feature_maps[key] = tf.layers.conv2d(feature, 
                    filters=channel_num, 
                    kernel_size=(3,3),
                    padding='same')
    return feature_maps
if __name__ == '__main__':
  main()

