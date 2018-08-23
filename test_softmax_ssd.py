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
from weakly_detection.utils import label_map_util

from PIL import Image
import matplotlib.pyplot as plt
slim = tf.contrib.slim

#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / ex.sum(axis=0)

def show_det_result(img,boxes):
    plt.imshow(np.squeeze(np.uint8(img)))
    bb_weight = boxes[2] - boxes[0]#min(image_weight-x1[cls_index], i_pool)
    bb_height = boxes[3] - boxes[1]#min(image_height-y1[cls_index], i_pool)
    plt.gca().add_patch(
            plt.Rectangle((boxes[0],boxes[1]),
                bb_weight,bb_height,
                fill=False,
                edgecolor='r', linewidth=3))
    plt.show()

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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

filenames = ['./pascal_train.record']
    
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_voc_tf)
#dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.repeat()
batch_size = 1
batched_dataset = dataset.batch(batch_size)

iterator = batched_dataset.make_one_shot_iterator()

images, labels = iterator.get_next()


def preprocess(images):
    images = tf.to_float(images)
    return images

images = preprocess(images)

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

def get_class_relate_feature(feature_maps, class_num=20):
    with tf.name_scope("class_relate_feature"):

        for key, feature in feature_maps.items():
            feature_maps[key] = tf.layers.conv2d(feature, 
                    filters=class_num, 
                    kernel_size=(3,3),
                    padding='same')
    return feature_maps

feature_maps = get_class_relate_feature(feature_maps)


#get result
class_relate_features = []
for key,feature in feature_maps.items():
    feature_shape = feature.get_shape()
    feature_shape.assert_has_rank(rank=4)
    class_relate_features.append(tf.reduce_sum(feature, [1,2]))
    
logits = tf.add_n(class_relate_features)
prob = tf.nn.softmax(logits)
#pred = tf.argmax(prob)
pred = tf.round(prob)

accu = tf.metrics.recall(labels,prob)

saver = tf.train.Saver()
restore_saver = tf.train.Saver()
save_path = "./log-softmax_loss/model.ckpt"

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())


label_to_id_dict = label_map_util.get_label_map_dict('./pascal_label_map.pbtxt')
id_to_label_dict = {v:k for k,v in label_to_id_dict.items()}

#labels = labels/ tf.sum(labels)

labels_sum = tf.reduce_sum(labels,axis=1)
labels = labels / tf.reshape(labels_sum,(-1,1))

with tf.Session() as sess:
    sess.run(init)
    #restore from vgg pretrained from ILSVRC
    #saver.restore(sess,'../../Experiments/detection/models/research/Experiments/weights/vgg_16.ckpt')
   # while(True):
   #     r_l = sess.run([labels])
   #     print(r_l)
    saver.restore(sess,save_path)
    count = 0
    image_height = 300
    image_weight = 300
    # conpare the label and the output
    while(False):
        r_l,r_p = sess.run([labels,prob])
        r_a = sess.run([accu])
        count+= 1
        if count % 100 == 0:
            print(r_l,r_p)
            print(r_a)

    #show detect
    while(True):
        count += 1 
        r_img, r_features,r_labels,r_probs ,r_logits = sess.run([images, feature_maps,labels,prob,logits])
        print("image:",count)
        print("labels:")
        for i,label in enumerate(r_labels[0]):
            if label > 0:
                print(id_to_label_dict[i+1])
        #C4: 2^3 = 8     #C7: 2^4 = 16      #C8: 2^5 = 32    #C9: 2^6 = 64    #C10: 2^7 = 128     #C11: 2^8 = 256

        #sort the probs 
        #get the index of features whose score larger than 0.1
        arg_probs= np.argsort(r_probs)[0][::-1]
        sort_probs = np.sort(r_probs)[0][::-1]
        keep_feature = []
        for i,s_prob in enumerate(sort_probs):
            if s_prob > 0.1:
                keep_feature.append(arg_probs[i])
        
        show_det_result(r_img,[0,0,0,0])

        for cls in keep_feature:
            cls_name = id_to_label_dict[cls+1]
            print(cls_name)
            #detection
            for i_pool ,key in enumerate(r_features):
                print("feature:",i_pool,key)
                feature = r_features[key] # 1 x H x W x C
                shape = feature.shape
                f_weight = shape[1]
                f_height = shape[2]
                f_classes = shape[3]
                feature_cls = feature[:,:,:,cls][0]
                plt.matshow(feature_cls,cmap='hot')
                plt.colorbar()
                plt.show()
                import pdb
                pdb.set_trace()
                
                continue 


                #show_det_result(feature[:,cls],[1,1,1,1])

                feature_flatten_cls = feature.reshape(-1, f_classes) # N x C, N = H x W
                np.sum(feature_flatten_cls,axis=0)
                feature_keep_cls = feature_flatten_cls[:,cls]
                feature_max_pos_per_cls = np.argmax(feature_keep_cls)
                #
                max_height = (feature_max_pos_per_cls / f_weight).astype(np.int32) #??should be height or weight??
                max_weight = feature_max_pos_per_cls - max_height * f_height 
                #get origin position
                i_pool = i_pool +3
                i_pool = 2**i_pool
                #get bbox
                x1 = max_weight * i_pool
                y1 = max_height * i_pool
                x2 = min(image_weight, x1 + i_pool)
                y2 = min(image_height, y1 + i_pool)
                bbox = [x1,y1,x2,y2]

                show_det_result(r_img,bbox)





    #r_image_features = sess.run([image_features])
#
def main():
  pass

if __name__ == '__main__':
  main()

