#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: models/losses.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月23日 星期四 09时58分33秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

 
import tensorflow as tf

def split_layer_softmax(cls_relate_features,labels):
  losses = []
  with tf.name_scope("losses"):
      for i,c_feature in enumerate(cls_relate_features):
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.to_float(labels), logits=c_feature, name='loss'))
          tf.summary.scalar("loss_{}".format(i),loss)
          losses.append(loss)
  return losses

def ranking_loss(cls_relate_features,labels):
    pass

def main():
  pass

if __name__ == '__main__':
  main()

