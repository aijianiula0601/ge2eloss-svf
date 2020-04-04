#!/bin/bash

set -ex

cd ../

meta_file='/home/huangjiahong/tmp/svf/dataset/kefu/record_wangjia_speakers_segment_feature_17to25_16kHZ/meta.txt'
log_dir='/home/huangjiahong/tmp/svf/training/kefu_wangjia/training/ge2e_tensorflow/log'
checkpoint_dir='/home/huangjiahong/tmp/svf/training/kefu_wangjia/training/ge2e_tensorflow/checkpoint'

python -u train.py \
  --train_meta_files $meta_file \
  --epochs 10000 \
  --max_keep_model 6000 \
  --batch_size 80 \
  --utterances_per_speaker 10 \
  --lr_decay_type constant \
  --learning_rate 0.001 \
  --min_learn_rate 0.0001 \
  --decay_rate 0.98 \
  --warmup_steps 1000 \
  --process_name sf_wj_17to25 \
  --gpu_devices 4,5,6,7 \
  --thread_num 10 \
  --buffer_size 10 \
  --log_dir $log_dir \
  --checkpoint $checkpoint_dir
