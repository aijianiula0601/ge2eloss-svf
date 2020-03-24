import os
import numpy as np
import tensorflow as tf

from model import SvfOjb
import hparams as hp
import audio as audio_ops
from helper.slice_utterance import slice_utterance_mel

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
model_file = "/home/huangjiahong/tmp/svf/better_model/20191211/dev_mode_epoch_191_gs_47368.0_eer_0.027632"
min_second_utterance = 3  # require min 3 second for audio


class SvfModel:

    def __init__(self, sess):

        self.model_file = model_file
        self.sess = sess

    def init_model(self):

        self.svf_model = SvfOjb()
        self.svf_model.build_inference()
        saver = tf.train.Saver(tf.global_variables())

        try:
            saver.restore(self.sess, self.model_file)
            # finalize model
            tf.get_default_graph().finalize()
            print("loading model done! model path:{}".format(self.model_file))
        except Exception as err:
            print('loading model failed!!!')
            print(err)
            pass

    def get_audios_embeds(self, audio_file):

        wav, sr = audio_ops.load_wav(audio_file)
        wav = np.array(wav)
        # audio preprocess
        wav = audio_ops.preprocess_wav(wav, source_sr=hp.sampling_rate)

        if len(wav) < min_second_utterance * hp.sampling_rate:
            print(' 音频有效长度({})小于指定最小长度({})'.format(len(wav) // hp.sampling_rate,
                                                   min_second_utterance))

        frames_batch = slice_utterance_mel(wav)  # shape=[batch_size, n_frames, n_channels]#对音频进行分割为多段
        [partial_embeds] = self.sess.run([self.svf_model.embeds],
                                         feed_dict={self.svf_model.inpt_inference: frames_batch})
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        return embed


if __name__ == '__main__':
    sess = tf.Session()
    svf_model = SvfModel(sess)

    svf_model.init_model()

    audio_file = '/home/huangjiahong/tmp/tts/dataset/api/baidu_tengxun_speakers/tengxun_0.wav'
    x = svf_model.get_audios_embeds(audio_file)

    print(np.shape(x))
