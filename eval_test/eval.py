import numpy as np
import tensorflow as tf

from model import SvfOjb
from params import pm
from dataset.test_data_load import get_dataset
from modules import calculate_eer

# ------------------------------------------------------
# 初始化模型
# ------------------------------------------------------

model_file = "E:\Download/dev_mode_epoch_1826_gs_142428.0_eer_0.016528"

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
# 计算测试集的eer
# ------------------------------------------------------

eer_cl = []
for step in range(num_batch * 5):
    feature = sess.run(next_element)

    [sim_matrix] = sess.run([svfObj.sim_matrix],
                            feed_dict={svfObj.inpt_inference: feature})

    eer_v = calculate_eer(sim_matrix)

    print('eer_v:', eer_v)
    eer_cl.append(eer_v)

print('==========avg_eer:', np.mean(np.array(eer_cl)))
