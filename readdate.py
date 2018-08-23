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

filenames = ['./tfrecords/voc_2007_train_016.tfrecord']
slim = tf.contrib.slim

def _parse_voc_tf(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(example_proto,items=keys)
    tensor_dict = dict(zip(keys,tensors))
    image = tensor_dict['image']
    shape = tensor_dict['shape']
    bboxes = tensor_dict['object/bbox']
    labels = tensor_dict['object/label']
    return image,shape,bboxes,labels

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_voc_tf)
iterator = dataset.make_one_shot_iterator()
with tf.Session() as sess:
  image,shape,bboxes,labels = iterator.get_next()
  print(sess.run([image,shape,bboxes,labels]))
  
def main():
  pass

if __name__ == '__main__':
  main()

