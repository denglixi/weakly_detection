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
from weakly_detection.feature_generator import multi_resolution_feature_maps
from weakly_detection.data_decoder import _parse_voc_tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint
from weakly_detection.net_utils import pad_to_multiple, conv_hyperarms_fn, filter_features, get_class_relate_feature

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

tf.logging.set_verbosity(tf.logging.INFO)

class MonitorRunHook(tf.train.SessionRunHook):
    def after_run(self,run_context,run_values):
        loss_value = run_values.results
        print(loss_value)

def SSD_Weakly_Model(
        features,
        labels,
        mode,
        params):

    images = features
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
    #training_hooks = []
    #training_hooks.append(tf.train.LoggingTensorHook([total_loss],20))
    return tf.estimator.EstimatorSpec(mode,loss=loss, train_op=train_op)#, training_hooks=training_hooks)




    #r_image_features = sess.run([image_features])
#

def dataset_input_fn():
    #Get data
    filenames = ['./pascal_train.record']
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_voc_tf)
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    batch_size = 16
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

def main():
    #get net
    MODEL_NAME = 'softmax_split_sigmoid'
    classifier = tf.estimator.Estimator(
            model_fn=SSD_Weakly_Model,
            model_dir='./results/{}/'.format(MODEL_NAME),
            )
    classifier.train(input_fn=dataset_input_fn)
            



if __name__ == '__main__':
  main()

