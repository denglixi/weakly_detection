#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: saver.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年08月24日 星期五 14时19分35秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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
def main():
  pass

if __name__ == '__main__':
  main()

