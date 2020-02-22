import numpy as np
import tensorflow as tf

from model import SvfOjb
import hparams as hp
import audio as audio_ops
from helper.slice_utterance import slice_utterance_mel
from service.config.logger import MyLogger

logger = MyLogger.__call__().get_logger()

# 模型位置
model_file = r'E:\dataset\svf\model\ge2e\20191211/dev_mode_epoch_191_gs_47368.0_eer_0.027632'
min_second_utterance = 3  # 音频的最小长度为3s


class SvfModel:

    def __init__(self, sess):

        self.model_file = model_file

        # 初始化模型
        self.sess = sess

    def init_model(self):

        self.svf_model = SvfOjb()
        self.svf_model.build_inference()
        saver = tf.train.Saver(tf.global_variables())

        try:
            saver.restore(self.sess, self.model_file)
            # 冻结模型
            tf.get_default_graph().finalize()
            logger.info("已加载训练好的模型:{}".format(self.model_file))
        except Exception as err:
            logger.error('无法加载训练好的模型!!!!')
            logger.error(err)
            pass

    def get_audios_embeds(self, sess, request_id, audio_body_dic):
        """
        获取音频文件的特征
        :param audio_body_dic: dic,格式为：{1:(wav,sr),2:{wav,sr},..}
        :return: 所有音频的embed向量
        """
        embed_result = {}

        for audio_id in audio_body_dic.keys():
            wav, sr = audio_body_dic[audio_id]
            wav = np.array(wav)
            # 预处理音频
            wav = audio_ops.preprocess_wav(wav, source_sr=hp.sampling_rate)

            if len(wav) < min_second_utterance * hp.sampling_rate:
                logger.info('request_id:{} audio_id:{} 音频有效长度({})小于指定最小长度({})'.format(request_id,
                                                                                      audio_id,
                                                                                      len(wav) // hp.sampling_rate,
                                                                                      min_second_utterance))

            frames_batch = slice_utterance_mel(wav)  # shape=[batch_size, n_frames, n_channels]#对音频进行分割为多段
            [partial_embeds] = sess.run([self.svf_model.embeds],
                                        feed_dict={self.svf_model.inpt_inference: frames_batch})
            raw_embed = np.mean(partial_embeds, axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)
            embed_result[audio_id] = embed.tolist()

        return embed_result


if __name__ == '__main__':
    sess = tf.Session()

    model_file = ''
    model = SvfModel(sess)

    model.init_model()

    audio_file = 'E:/Download/1.wav'
    body = audio_ops.load_wav(audio_file)
    body1 = audio_ops.load_wav(audio_file)

    bodys = [body, body1]

    x = model.get_audios_embeds(bodys)

    print(x)
