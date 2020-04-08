#!/bin/bash

set -ex

######################
#对数据进行采样
######################

#read_dir中存储的是每个speaker的音频，每个speaker一个文件夹
read_dir='/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16'
save_sample_dir='/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16_sample'

#指定只需要3s以上的音频
min_second=3
#每个speaker抽样音频个数
sample_num=10

python -u sample_kefu.py $read_dir $save_sample_dir $min_second $sample_num


######################
#提取audio embedding 向量
######################

save_embedding_dir='/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16_embed_vector_v2'
#训练声纹识别之后采用的数据集目录，如果该变量为空，则不检测
training_speaker_dir='/home/user/tmp/svf/dataset/kefu/record_wangjia_speakers_segment_feature_17to30'


python -u utterance_vector.py $save_sample_dir $save_embedding_dir $training_speaker_dir