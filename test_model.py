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
from weakly_detection.dataset.pascal_voc import get_input_fn
from weakly_detection.models.model_factory import get_model_fn
from weakly_detection.utils import label_map_util
from PIL import Image
import matplotlib.pyplot as plt

slim = tf.contrib.slim

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tf.logging.set_verbosity(tf.logging.INFO)

def show_det_result(img,boxes):
    plt.imshow(np.squeeze(np.uint8(img)))
    bb_weight = boxes[2] - boxes[0]#min(image_weight-x1[cls_index], i_pool)
    bb_height = boxes[3] - boxes[1]#min(image_height-y1[cls_index], i_pool)
    plt.gca().add_patch(
            plt.Rectangle((boxes[0],boxes[1]),
                bb_weight,bb_height,
                fill=False,
                edgecolor='r', linewidth=3))
    plt.show()

def main():
    #get net
    label_to_id_dict = label_map_util.get_label_map_dict('./pascal_label_map.pbtxt')
    id_to_label_dict = {v:k for k,v in label_to_id_dict.items()}
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000, 
                                        keep_checkpoint_max=0)

    train_input_fn = get_input_fn("./pascal_train.record",16,True)
    eval_input_fn = get_input_fn("./pascal_val.record",16,True)
    MODEL_NAME = 'softmax_gap'
    #MODEL_NAME = 'softmax_split'
    model_fn = get_model_fn(MODEL_NAME)
    training_params = {}
    training_params["restore_from_pretrain_weight"] = True
    training_params["pretrain_weight_path"] = '../../Experiments/detection/models/research/Experiments/weights/vgg_16.ckpt'
    #training_params["pretrain_weight_path"] = "./results/softmax_conv_cls_feature/model.ckpt-275000"
    training_params["threshold_T"] = 0.1
    training_params["lr"] = 0.001

    eval_params = {}   
    eval_params["threshold_T"] = 0.1
    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir='./results/estimator_{}/'.format(MODEL_NAME),
            params=training_params,
            config=run_config
            )
    
    features_key = [ 'conv4_3', 'fc7', 'fc7_2_Conv2d_2_3x3_s2_512', 'fc7_2_Conv2d_3_3x3_s2_256', 'fc7_2_Conv2d_4_3x3_s2_256', 'fc7_2_Conv2d_5_3x3_s2_256' ]
    for pred in estimator.predict(input_fn=eval_input_fn):
        image = pred['image']

        show_det_result(image,[0,0,0,0])

        probs = []
        for i in range(6):
            probs.append(pred['prob_{}'.format(i)])

        for i_pool,key in enumerate(features_key):
            print("layer_key:{}".format(key))
            arg_probs= np.argsort(probs[i_pool])[::-1]
            sort_probs = np.sort(probs[i_pool])[::-1]
            keep_feature = []
            for i,s_prob in enumerate(sort_probs):
                if s_prob > 0.5:
                    keep_feature.append(arg_probs[i])
                else:
                    break
            for cls in keep_feature:
                cls_name = id_to_label_dict[cls+1]
                print("class:",cls_name)
                # 2. show class channel data in feature
                feature = pred[key] # 1 x H x W x C
                shape = feature.shape
                f_weight = shape[0]
                f_height = shape[1]
                f_classes = shape[2]
                feature_cls = feature[:,:,cls]
                plt.matshow(feature_cls,cmap='hot')
                plt.colorbar()
                plt.show()









if __name__ == '__main__':
  main()

