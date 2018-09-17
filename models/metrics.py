#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: metrics.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月24日 星期五 14时18分05秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
def set_metrics(preds,labels):
    metrics_ops = {}
    with tf.name_scope("recall"):
        for i,pred in enumerate(preds):
            recall = tf.metrics.recall(labels,pred)
            metrics_ops["recall_{}".format(i)] = recall
            tf.summary.scalar("recall_layer_{}".format(i),recall[0])

    with tf.name_scope("accuracy"):
        for i,pred in enumerate(preds):
            recall = tf.metrics.recall(labels,pred)
            accu = tf.metrics.accuracy(labels,pred)
            metrics_ops["accu_{}".format(i)] = accu
            tf.summary.scalar("accu_{}".format(i),accu[0])

    with tf.name_scope("precision"):
        for i,pred in enumerate(preds):
            recall = tf.metrics.recall(labels,pred)
            precision  = tf.metrics.precision(labels,pred)
            metrics_ops["precision_{}".format(i)] = precision
            tf.summary.scalar("precision_{}".format(i),precision[0])
    return metrics_ops
def main():
  pass

if __name__ == '__main__':
  main()

