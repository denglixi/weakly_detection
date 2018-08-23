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
from weakly_detection.dataset.get_data import get_data
from weakly_detection.net_utils.net_utils import pad_to_multiple,conv_hyperarms_fn, filter_features, dict_feature_conv_to_cls
slim = tf.contrib.slim
from weakly_detection.models.SSD_backbone import SSD_Backbone_Net

from weakly_detection.models.cls_relate_feature import cls_relate_feature_conv
import weakly_detection.models.losses as losses
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

save_dir = "./results/softmax_conv_cls_feature/"
restore_path = './results/softmax_conv_cls_feature/model.ckpt-290000'

train_filenames = ['./pascal_train.record']
val_filenames = ['./pascal_train.record']
images,labels = get_data(train_filenames,is_shuffled=True)
val_images, val_labels = get_data(val_filenames,batch_size=1)
    
# normalize the label for softmax
labels_sum = tf.reduce_sum(labels,axis=1)
labels = labels / tf.reshape(labels_sum,(-1,1))

def preprocess(images):
    images = tf.to_float(images)
    return images

images = preprocess(images)
val_images = preprocess(val_images)


def get_model(input_image):
    feature_maps = SSD_Backbone_Net(images)
    feature_maps = dict_feature_conv_to_cls(feature_maps,channel_num=256)
    cls_relate_features = cls_relate_feature_conv(feature_maps,cls_num=20) # shape of output is same with labels. It is logits
    return cls_relate_features
# construct training model 
cls_relate_features = get_model(images)
losses = losses.split_layer_softmax(cls_relate_features,labels)
total_loss = tf.add_n(losses)

#construct evaluation model
val_logits = get_model(val_images)
tf.summary.scalar("total_loss",total_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
global_step_tensor = tf.Variable(0, trainable=False, name='global_step',dtype=tf.int64)
train_op = optimizer.minimize(total_loss,global_step=global_step_tensor)
merge_summary = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=1000)
model_save_path = os.path.join(save_dir,'model.ckpt')
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
    train_writer = tf.summary.FileWriter(save_dir,sess.graph)
    # restore from vgg pretrained from ILSVRC
    #restore_saver.restore(sess,vgg_weight_path)
    saver.restore(sess,restore_path)
    count = tf.train.global_step(sess,global_step_tensor)
    print("begin count:",count)
    while(True):
        count += 1 
        _, r_loss, r_prob, r_labels,r_summary = sess.run([train_op, total_loss, prob, labels, merge_summary])
        train_writer.add_summary(r_summary,tf.train.global_step(sess,global_step_tensor))

        if count % 100 == 0:
            print("total_loss:",np.mean(r_loss))
            #print("label:",r_labels)
            #print("prob:",r_prob)
            if count % 5000 == 0:
                print("Model save in %s" % saver.save(sess,model_save_path,global_step=tf.train.global_step(sess,global_step_tensor)))

    #r_image_features = sess.run([image_features])
#
def main():
  pass

if __name__ == '__main__':
  main()

