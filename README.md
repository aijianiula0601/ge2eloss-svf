# ge2eloss-svf
本项目是采用tensorflow实现Google的ge2e的声纹识别项目,论文：https://arxiv.org/abs/1710.10467

# 环境
python3.7
tensorflow-gpu==1.4.0

# 数据集
采用了我们内部的数据集，音频的采样率为8k，是电话客服中的数据。根据用户打进来的电话号码划分不同的人。共收集了3万个用户的音频，每个用户音频采用活性检测后分割为更小的片段，保证每段音频至少有6s。

数据准备可以参考代码：dataset/generate_meta.py

# 训练命令

如下是我的训练命令，可参考下
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
# 验证eer

可以参考代码中：
```
1.eval_test/v2/prepare_eval_data/test_data_utterance_vector.py
2.eval_test/v2/eval_eer.py
```

# 训练效果
训练收敛情况：
![image]https://github.com/aijianiula0601/ge2eloss-svf/blob/master/imgs/ge2e_train.jpg

测试集上的效果：

| 数据集-音频长度5s以上 | 不在训练集中的240个speaker,每个speaker至少10段5s以上音频,采样率8k | aishell的test数据集，采样率16k转为8k才测试 | aishell的train数据集,采样率16k转为8k才测试 | magicdata_mandarin_chinese采样率16k转为8k才测试 |
| ------ | ------ | ------ | ------ | ------ |
| eer | 0.018 | 0.04 | 0.025 | 0.04 |
