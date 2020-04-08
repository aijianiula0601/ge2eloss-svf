from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
import os
import sys
import shutil

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))))
print('prj_dir:', prj_dir)
sys.path.append(prj_dir)

from model import SvfOjb
import audio
from params import pm
import hparams as hp
from helper.slice_utterance import slice_utterance_mel

"""
功能：获取一段音频的embedding 向量
"""

# ------------------------------------------------------
# 初始化模型
# ------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# model_file = "/home/user/tmp/svf/checkpoint_wj/min_dev/dev_mode_epoch_253_gs_min_78430.0_eer_0.027210"
model_file = "/home/user/tmp/svf/checkpoint_wj_20191211/dev_mode_epoch_281_gs_69688.0_eer_0.016842"

sess = tf.Session()
# 初始化模型
svfObj = SvfOjb()

# 初始化推理图
svfObj.build_inference()
# 保存模型
saver = tf.train.Saver(tf.global_variables(), max_to_keep=pm.max_keep_model)
try:
    saver.restore(sess, model_file)

    print('已加载最近训练模型!')
except Exception as e:
    print("无法加载旧模型,训练过程会覆盖旧模型!")
    pass

# 冻结模型
tf.get_default_graph().finalize()

min_second_utterances = 3  # 限制音频最小长度


def embedding_wav_and_save_vector(frames_batch, save_path):
    """
    保存音频的embedding向量
    """

    [partial_embeds] = sess.run([svfObj.embeds], feed_dict={svfObj.inpt_inference: frames_batch})
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    np.save(file=save_path, arr=embed)


if __name__ == '__main__':

    read_dir = sys.argv[1]
    save_dir = sys.argv[2]

    train_speaker_dir = sys.argv[3]
    train_speaker_set = set()
    if train_speaker_dir is None or train_speaker_dir == "":
        for speaker_dir in Path(train_speaker_dir).glob("*"):
            train_speaker_set.add(speaker_dir.name)

    print('训练speaker个数:{}'.format(len(train_speaker_set)))

    # 新建保存向量的目录
    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(exist_ok=True)

    i = 0
    for speaker_dir in Path(read_dir).glob("*"):
        if speaker_dir.name in train_speaker_set:  # 只有不出现在训练集的speaker才拿来做测试
            print('{} 存在训练集中，丢弃！！'.format(speaker_dir.name))
            continue

        for wav_path in speaker_dir.glob("*.wav"):
            save_speaker_dir = Path(save_dir).joinpath(speaker_dir.name)
            save_speaker_dir.mkdir(exist_ok=True)

            # 预处理音频
            wav = audio.preprocess_wav(wav_path, source_sr=hp.sampling_rate)
            if len(wav) < min_second_utterances * hp.sampling_rate:
                continue

            frames_batch = slice_utterance_mel(wav)  # shape=[batch_size, n_frames, n_channels]#对音频进行分割为多段

            save_wav_path = str(save_speaker_dir.joinpath(wav_path.name.replace(".wav", "_{}.npy".format(len(wav)))))
            # 获取音频的embedding向量，然后保存
            embedding_wav_and_save_vector(frames_batch, save_wav_path)

        i += 1
        if i % 100 == 0:
            print(i)
