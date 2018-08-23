#!/usr/bin/env python
#-*-coding:utf-8-*-
#########################################################################
#    > File Name: readdate.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com 
#    > Created Time: 2018年03月28日 星期三 10时24分36秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

filenames = ['./pascal_train.record']
slim = tf.contrib.slim

def _parse_voc_tf(example_proto):
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(example_proto,items=keys)
    tensor_dict = dict(zip(keys,tensors))
    image = tensor_dict['image']
    #resize image
    image = tf.expand_dims(image,0)
    image = tf.image.resize_bicubic(image,(300,300),name='image/resize')
    image = tf.squeeze(image)
    labels = tensor_dict['object/label']
    return image,labels

def test():
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_voc_tf)
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
      while(True):
        image,labels = iterator.get_next()
        print(sess.run([labels]))
  
def main():
  test()
  pass

if __name__ == '__main__':
  main()

