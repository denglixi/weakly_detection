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
from weakly_detection.feature_generator import multi_resolution_feature_maps
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint

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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def preprocess(images):
    images = tf.to_float(images)
    return images

class SSD_meta():
    def __init__(self):
        self.base_net = self.base_net()
def ssd_model(image,label)
    image = preprocess(image)
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
    
    
    feature_maps = get_class_relate_feature(feature_maps)
    
        
    class_relate_features = []
    for key,feature in feature_maps.items():
        feature_shape = feature.get_shape()
        feature_shape.assert_has_rank(rank=4)
        class_relate_features.append(tf.reduce_sum(feature, [1,2]))
        
    logits = tf.add_n(class_relate_features)
    prob = tf.sigmoid(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels), logits=logits, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(loss)

saver = tf.train.Saver()
save_path = "./log-finetune/model.ckpt"
#restroe from vgg
vgg_weight_path = '../../Experiments/detection/models/research/Experiments/weights/vgg_16.ckpt'
  #find available_var in vgg weight
var_map = {}
for variable in tf.global_variables():
    var_name = variable.op.name
    var_map[var_name] = variable
available_var_map = get_variables_available_in_checkpoint(var_map,vgg_weight_path)
restore_saver = tf.train.Saver(available_var_map)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #restore from vgg pretrained from ILSVRC
    restore_saver.restore(sess,vgg_weight_path)
    count = 0
    while(True):
        count += 1 
        _, r_loss, r_prob, r_labels = sess.run([train_op, loss, prob, labels])
        if count % 100 == 0:
            print(np.mean(r_loss))
            #print("label:",r_labels)
            #print("prob:",r_prob)
            if count % 2000 == 0:
                print("Model save in %s" % saver.save(sess,save_path))

    #r_image_features = sess.run([image_features])
#
def main():
  pass

if __name__ == '__main__':
  main()

