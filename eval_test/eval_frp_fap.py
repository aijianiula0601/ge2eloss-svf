import numpy as np
import tensorflow as tf

from model import SvfOjb
from params import pm
from dataset.test_data_load import get_dataset
from eval_test import frp_fap

# ------------------------------------------------------
# 初始化模型
# ------------------------------------------------------

model_file = "E:\Download/dev_mode_epoch_407_gs_31746.0_eer_0.030099"

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

# mel_feature_dir = 'E:/dataset/TTS/dataset/test-clean/LibriSpeech/test-clean_features'
mel_feature_dir = r'E:\dataset\TTS\dataset\aishell_test_features'
meta_file = mel_feature_dir + '/meta.txt'
dataset, num_batch = get_dataset(mel_feature_base_dir=mel_feature_dir, meta_file=meta_file)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess.run(iterator.initializer)

# 冻结模型
tf.get_default_graph().finalize()

# ------------------------------------------------------
# 计算错误接受率，错误拒绝率
# ------------------------------------------------------

threshold_p = 0.8
fap_cl = []
frp_cl = []
for step in range(num_batch * 5):
    feature = sess.run(next_element)

    # embeds shape=[speakers_per_batch,utterances_per_speaker,embed_size]
    [embeds] = sess.run([svfObj.embeds],
                        feed_dict={svfObj.inpt_inference: feature})

    # 计算错误拒绝率
    for i in range(np.shape(embeds)[0]):
        sub_embed = embeds[i]

        intra_class_matrix = np.matmul(sub_embed, sub_embed.T)

        frp = frp_fap.frp(intra_class_matrix, threshold_p=threshold_p)

        print("frp:", frp)
        frp_cl.append(frp)

    # 计算错误接受率
    fap_embeds = embeds.swapaxes(0, 1)  # shape=[utterances_per_speaker,speakers_per_batch,embed_size]
    for i in range(np.shape(fap_embeds)[0]):
        sub_embed = fap_embeds[i]

        inter_class_matrix = np.matmul(sub_embed, sub_embed.T)

        fap = frp_fap.fap(inter_class_matrix, threshold_p=threshold_p)

        print("frp:", fap)
        fap_cl.append(fap)

print("平均错误拒绝率:{}，平均错误接受率:{}".format(np.mean(np.array(frp_cl)), np.mean(np.array(fap_cl))))
