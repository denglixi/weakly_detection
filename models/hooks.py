#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: hooks.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月23日 星期四 14时20分34秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from weakly_detection.utils.variables_helper import get_variables_available_in_checkpoint

def get_variable_vgg(model_path):
  #restroe from vgg
  vgg_weight_path = model_path
  #find available_var in vgg weight
  var_map = {}
  for variable in tf.global_variables():
      var_name = variable.op.name
      var_map[var_name] = variable
  available_var_map = get_variables_available_in_checkpoint(var_map,vgg_weight_path)
  return available_var_map


class RestoreHook(tf.train.SessionRunHook):
  def __init__(self, init_fn):
    self.init_fn = init_fn
  def afterc_create_session(self,session, coord=None):
    if session.run(tf.train.get_or_create_global_step()) == 0:
      self.init_fn(session)

def main():
  pass

if __name__ == '__main__':
  main()

