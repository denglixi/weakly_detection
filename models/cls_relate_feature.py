#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: models/cls_relate_feature.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月23日 星期四 09时45分16秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def main():
  pass

def cls_relate_feature_conv(feature_maps,cls_num=20):
    #use convolution resize feature to [b, 1, 1, class_num] 
    class_relate_features = []
    for key,feature in feature_maps.items():
        feature_shape = feature.get_shape()
        feature_shape.assert_has_rank(rank=4)
        feature_shape_list = feature_shape.as_list()
        cls_relate_feature = tf.layers.conv2d(feature,
                filters=cls_num,
                kernel_size=(feature_shape_list[1],feature_shape_list[2]),
                padding='valid')
        class_relate_features.append(tf.squeeze(cls_relate_feature,axis =[1,2]))
    return class_relate_features
if __name__ == '__main__':
  main()

