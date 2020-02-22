from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
import os
import sys
import shutil


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

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# 限定为在cpu上运行
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_file = "/home/huangjiahong/tmp/svf/better_model/20191211/dev_mode_epoch_191_gs_47368.0_eer_0.027632"

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
    # fp = 'E:\Download/A7_4.wav'
    # save_path = 'E:\Download/aa.npy'
    # embedding_wav_and_save_vector(fp, save_path)

    read_dir = '/home/huangjiahong/tmp/common_dataset/data_aishell/wav/test'
    save_dir = '/home/huangjiahong/tmp/svf/dataset/kefu/test_dataset/test_aishell_embed_vector_v1'

    # 新建保存向量的目录
    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(exist_ok=True)

    i = 0
    for speaker_dir in Path(read_dir).glob("*"):

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
