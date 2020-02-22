# ge2eloss-svf
Tensorflow implementation of Generalized End-to-End Loss for speaker verification, proposed by google in 2017 in https://arxiv.org/pdf/1710.10467.pdf

# Environment
python3.7,tensorflow-gpu==1.4.0

# Dataset
We use our own dataset which belong to the game customer service.The audio sampled with 8k.We collected the audios with 30000 speakers and segement the audio to short but greater than 6s with Vad technology.

To preprocess the data with dataset/generate_meta.py

# Training script

```bash
python train.py \
--train_meta_files \
/tmp/meta.txt \
--epochs 10000 \
--max_keep_model 100 \
--batch_size 100 \
--utterances_per_speaker 10 \
--lr_decay_type constant \
--process_name sf_wj \
--gpu_devices 0,1,2,3,4 \
--thread_num 10 \
--buffer_size 10 \
--log_dir /tmp/svf/log_wj \
--checkpoint /tmp/svf/checkpoint_wj
```
# Training result

<div align=center>
  <img src="https://github.com/aijianiula0601/ge2eloss-svf/blob/master/imgs/ge2e_train.jpg"  alt="训练展示" width = "300" height = "300" />
</div>

# Calculate EER

```
1.eval_test/v2/prepare_eval_data/test_data_utterance_vector.py
2.eval_test/v2/eval_eer.py
```

# Test result

| dataset-audio>=5s | test-data with 240 speakers not found in train-data.Each speaker must have at least 10 pieces of 5s + audio.Sample rate is 8k | testdata from aishell.Resample to 8k | traindata in aishell.Resample to 8k | magicdata_mandarin_chinese's test data resample to 8k |
| ------ | ------ | ------ | ------ | ------ |
| eer | 0.018 | 0.04 | 0.025 | 0.04 |

# Reference

https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder
