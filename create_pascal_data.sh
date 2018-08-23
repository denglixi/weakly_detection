#!/bin/bash
#########################################################################
# File Name: create_pascal_data.sh
# Author: Deng Lixi
# mail: 285310651@qq.com
# Created Time: 2018年06月28日 星期四 14时58分05秒
#########################################################################

python ./create_pascal_tf_record.py \
        --label_map_path=./pascal_label_map.pbtxt \
            --data_dir=/home/dlx/data/VOCdevkit --year=VOC2012 --set=val \
                --output_path=pascal_val.record
