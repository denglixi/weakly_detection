#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: model_factory.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月17日 星期五 16时06分53秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from weakly_detection.models.wsd_softmax_split import wsd_softmax_split_model
from weakly_detection.models.wsd_softmax_split_sigmoid import wsd_softmax_split_sigmoid_model 
from weakly_detection.models.wsd_softmax_conv import wsd_softmax_conv_model_fn 
from weakly_detection.models.wsd_softmax_gap import wsd_softmax_gap_model_fn

def get_model_fn(model_name):
  if model_name == 'softmax_split':
    return wsd_softmax_split_model 
  if model_name == 'softmax_split_sigmoid':
    return wsd_softmax_split_sigmoid_model 
  if model_name == 'softmax_conv':
    return wsd_softmax_conv_model_fn
  if model_name == 'softmax_gap':
    return wsd_softmax_gap_model_fn
  
def main():
  pass

if __name__ == '__main__':
  main()

