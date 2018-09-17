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
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint
from weakly_detection.net_utils.net_utils import pad_to_multiple,conv_hyperarms_fn, filter_features, dict_feature_conv_to_cnls
from weakly_detection.models.SSD_backbone import SSD_Backbone_Net
from weakly_detection.models.cls_relate_feature import cls_relate_feature_conv
from weakly_detection.models.losses import split_layer_softmax 

#from tensorflow.contrib.framework import assign_from_checkpoint_fn
from weakly_detection.models.hooks import RestoreHook, get_variable_vgg
slim = tf.contrib.slim


import os
def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
  """Returns a function that assigns specific variables from a checkpoint.
  If ignore_missing_vars is True and no variables are found in the checkpoint
  it returns None.
  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
        use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
        checkpoint to the corresponding variables to initialize. If empty or
        `None`, it would return `no_op(), None`.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
        the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
        which are of different shape then the ones stored in the checkpoint but
        which have the same number of elements.
  Returns:
    A function that takes a single argument, a `tf.Session`, that applies the
    assignment operation. If no matching variables were found in the checkpoint
    then `None` is returned.
  Raises:
    ValueError: If var_list is empty.
  """
  if isinstance(var_list, list):
    variable_names_map = {variable.op.name: variable for variable in var_list}
  elif isinstance(var_list, dict):
    variable_names_map = var_list 
  else:
    raise ValueError('`variables` is expected to be a list or dict.')
  ckpt_reader = tf.train.NewCheckpointReader(model_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  #if not include_global_step:
  #  ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
        vars_in_ckpt[variable_name] = variable
      else:
        tf.logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable.',
                        variable_name)
    else:
      tf.logging.warning('Variable [%s] is not available in checkpoint',
                      variable_name)
  #if isinstance(var_list, list):
  #  var_list = vars_in_ckpt.values()
  #else:
  #  var_list = vars_in_ckpt
  

  if vars_in_ckpt:
    saver = tf.train.Saver(vars_in_ckpt, reshape=reshape_variables)
    def callback(session):
      tf.logging.info("model restore from {}".format(model_path))
      saver.restore(session, model_path)
    return callback
  else:
    tf.logging.warning('No Variables to restore')
    return None
#model fn
def wsd_softmax_conv_model_fn(
        features,
        labels,
        mode,
        params):

    feature_maps = SSD_Backbone_Net(features)

    with tf.variable_scope("wsd_feature2cls"):
        #chang feature map to batch * N * N *channel 
        feature_maps = dict_feature_conv_to_cnls(feature_maps,channel_num=256)
        #chang feature map to batch * class_num 
        cls_relate_features = cls_relate_feature_conv(feature_maps,cls_num=20) # shape of output is same with labels. It is logits
    
    probs = []
    preds = []
    for c_feature in cls_relate_features:
        prob = tf.nn.softmax(c_feature)
        probs.append(prob)
        pred = tf.to_int32(prob > params['threshold_T'])
        preds.append(pred)

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

    losses = split_layer_softmax(cls_relate_features,labels)
    total_loss = tf.add_n(losses)
    tf.summary.scalar("total_loss",total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    #accuracy = tf.metrics.accuracy(labels,)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                loss=total_loss,
                eval_metric_ops=metrics_ops)
        pass

    #if mode == tf.estimator.ModeKeys.TRAIN:
    assert mode == tf.estimator.ModeKeys.TRAIN

    training_hooks = []

    #resotre from vgg
    if params!=None and 'restore_from_pretrain_weight' in params and params['restore_from_pretrain_weight']:
        weight_path = params['pretrain_weight_path']
        var_list = tf.global_variables()
        init_fn = assign_from_checkpoint_fn(weight_path,var_list,ignore_missing_vars=True)
        training_hooks.append(RestoreHook(init_fn))

    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,
            loss=total_loss, 
            train_op=train_op,
            eval_metric_ops=metrics_ops,
            training_hooks=training_hooks)


def main():
    #get net
    pass 



if __name__ == '__main__':
  main()

