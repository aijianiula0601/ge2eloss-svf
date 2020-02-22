import numpy as np
import tensorflow as tf

from model import SvfOjb
from params import pm
from dataset.test_data_load import load_one_person_m_utterances
from eval_test import frp_fap
from pathlib import Path

# ------------------------------------------------------
# 初始化模型
# ------------------------------------------------------

model_file = "E:\Download/dev_mode_epoch_1826_gs_142428.0_eer_0.016528"

pm.batch_size = 1
pm.gpu_devices = "1"
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

speakers_dir = r"E:\dataset\TTS\dataset\aishell_test_wavs"

# ------------------------------------------------------
# 计算错误接受率,采用一段语音分割后，计算每段的embed再去中心向量作为该段的向量，注意：frp和fap的模型位置要一样
# ------------------------------------------------------

threshold_p = 0.78
frp_cl = []

for speaker_dir in Path(speakers_dir).glob("*"):

    # speaker_utterances_mels的格式为：
    # [
    #     [utt_part1,n_frames,n_mels],
    #     [utt_part2,n_frames,n_mels],
    #     ...
    # ]
    speaker_utterances_mels = load_one_person_m_utterances(speaker_dir=speaker_dir, utterances_num=10, extension="wav")

    same_speaker_utterance_embeds_cl = []
    for utterance_mel in speaker_utterances_mels:
        # inpt,shape=[1,bsz1,n_frames,n_mels]
        feature = np.array([utterance_mel])

        # embeds shape=[1,utt_part1,embed_size]
        [embeds] = sess.run([svfObj.embeds],
                            feed_dict={svfObj.inpt_inference: feature})

        raw_embed = np.mean(embeds[0], axis=0)  # 计算中心向量
        utterance_embed = raw_embed / np.linalg.norm(raw_embed, 2)  # 规范化范式

        same_speaker_utterance_embeds_cl.append(utterance_embed)

    speaker_utterances_embeds = np.array(same_speaker_utterance_embeds_cl)

    # 计算错误拒绝率
    intra_class_matrix = np.matmul(speaker_utterances_embeds, speaker_utterances_embeds.T)

    frp = frp_fap.frp(intra_class_matrix, threshold_p=threshold_p)

    print("frp:", frp)
    frp_cl.append(frp)

print("平均错误拒绝率:{}".format(np.mean(np.array(frp_cl))))
