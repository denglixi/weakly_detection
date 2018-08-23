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
from weakly_detection.utils import label_map_util
from weakly_detection.net_utils.net_utils import pad_to_multiple,conv_hyperarms_fn, filter_features, get_class_relate_feature
from weakly_detection.dataset.get_data import get_data

from PIL import Image
import matplotlib.pyplot as plt
slim = tf.contrib.slim




import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

save_path = "./results/softmax_conv_cls_feature/model.ckpt-100000"

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


train_filenames = ['./pascal_train.record']
val_filenames = ['./pascal_train.record']
images,lables = get_data(train_filenames)
val_images, val_labels = get_data(val_filenames)


def preprocess(images):
    images = tf.to_float(images)
    return images

images = preprocess(images)
val_images = preprocess(val_images)

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

feature_maps = get_class_relate_feature(feature_maps,class_num=256)



#get result
class_relate_features = []
for key,feature in feature_maps.items():
    feature_shape = feature.get_shape()
    feature_shape.assert_has_rank(rank=4)
    feature_shape_list = feature_shape.as_list()
    cls_relate_feature = tf.layers.conv2d(feature,
            filters=20,
            kernel_size=(feature_shape_list[1],feature_shape_list[2]),
            padding='valid')
    class_relate_features.append(cls_relate_feature)


probs = []
preds = []
losses = []
for c_feature in class_relate_features:
    prob = tf.squeeze(tf.nn.softmax(c_feature),axis=[1,2])
    probs.append(prob)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_float(labels), logits=c_feature, name='loss'))
    losses.append(loss)
    #pred = tf.round(prob)
    #pred = tf.cond(prob > 0.1, lambda: 1, lambda: 0)
    pred = tf.to_int32(prob > 0.1)
    preds.append(pred)
    #result = tf.equal(tf.round)
total_loss = tf.add_n(losses)
    
#logits = tf.add_n(class_relate_features)
#pred = tf.argmax(prob)
#pred = tf.round(prob)

#accu = tf.metrics.recall(labels,prob)

saver = tf.train.Saver()
restore_saver = tf.train.Saver()


label_to_id_dict = label_map_util.get_label_map_dict('./pascal_label_map.pbtxt')
id_to_label_dict = {v:k for k,v in label_to_id_dict.items()}

#labels = labels/ tf.sum(labels)
labels_sum = tf.reduce_sum(labels,axis=1)
normed_labels = labels / tf.reshape(labels_sum,(-1,1))
recalls = []
precision = []
accuracies = []

for pred in preds:
    recalls.append(tf.metrics.recall(labels,pred))
    accuracies.append(tf.metrics.accuracy(labels,pred)) 
    precision.append(tf.metrics.precision(labels,pred)) 

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session() as sess:
    cls_statics = np.array([0 for _ in range(20)])
    label_statics = np.array([0 for _ in range(20)])
    sess.run(init)
    count = 0
    saver.restore(sess,save_path)
    print("model restore from {}".format(save_path))

    batch_loss = 0
    while(False):
        count+= 1
        r_total_loss,r_l,r_loss,r_probs,r_preds,r_r,r_a,r_precision = sess.run([total_loss, labels,losses,probs,preds,recalls,accuracies,precision])
        batch_loss += r_total_loss
        label_statics += r_l[0]
        one_cls_statics = np.array([0 for _ in range(20)])
        for pred in r_preds:
            one_cls_statics += pred[0]
        cls_statics += np.int32(one_cls_statics > 0)
        if count % 100 == 0:
            batch_loss /= 100
            print("batch loss:",batch_loss)
            batch_loss = 0
            print("loss:",r_loss)
            print("total loss:",r_total_loss)
            print("recall:",r_r)
            print("accu:",r_a)
            print("precision:",r_precision)
            print("cls:",cls_statics)
            print("label:",label_statics)
            print("\n\n------------------------------")

    #show detect
    while(True):
        count += 1 
        b_r_img, b_r_features,b_r_labels,b_r_probs = sess.run([images, feature_maps,labels,probs])
        
        for batch_i in range(batch_size):
            r_img = b_r_img[batch_i]
            #r_features = b_r_features[batch_i]
            r_labels = b_r_labels[batch_i]
            #r_probs = b_r_probs[0][batch_i]


            print("image:",count)
            print("labels:")
            for i,label in enumerate(r_labels):
                if label > 0:
                    print(id_to_label_dict[i+1])
            #C4: 2^3 = 8     #C7: 2^4 = 16      #C8: 2^5 = 32    #C9: 2^6 = 64    #C10: 2^7 = 128     #C11: 2^8 = 256

            #sort the probs 
            #get the index of features whose score larger than 0.1


            
            show_det_result(r_img,[0,0,0,0])

                #detection
            for i_pool ,key in enumerate(b_r_features):
                print("feature:",i_pool,key)
                #1. get cls in this feature!
                arg_probs= np.argsort(b_r_probs[i_pool][batch_i])[::-1]
                sort_probs = np.sort(b_r_probs[i_pool][batch_i])[::-1]
                keep_feature = []
                for i,s_prob in enumerate(sort_probs):
                    if s_prob > 0.1:
                        keep_feature.append(arg_probs[i])
                for cls in keep_feature:
                    cls_name = id_to_label_dict[cls+1]
                    cls_statics[cls] += 1
                    print("class:",cls_name)

                    # 2. show class channel data in feature
                    feature = b_r_features[key][batch_i] # 1 x H x W x C
                    shape = feature.shape
                    f_weight = shape[0]
                    f_height = shape[1]
                    f_classes = shape[2]
                    feature_cls = feature[:,:,cls]
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

