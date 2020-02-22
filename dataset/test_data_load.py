import os
import sys
import random
import numpy as np
import tensorflow as tf
import logging
from keras.preprocessing import sequence
from pathlib import Path

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print("---prj_dir:", prj_dir)
sys.path.append(prj_dir)
from params import pm
import hparams as hp
from helper.slice_utterance import slice_utterance_mel
import audio


def load_data(base_dir, meta_file):
    """
    load the meta data for training
    The meta data which contains in meta.txt.The lines like: speaker_id|mel_file_paths
    The mel_file_paths is the speark_id's m utterances,which segmented by comma,
        e.g.: The spearker_id is 1 and m=3,which line like: 1|mel1.npy,mel2.npy,mel3.npy
    :param base_dir: The directory contains the feature files
    :param meta_file: The file contain the meta data
    :return: The list contains the mel_file,exclude the speaker_id

    ps：The shape of mel.npy is same,which is [n_frames,n_mels]
    """

    rows_cl = []
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.replace("\n", "").split("|")
            features_files = parts[1]
            for fn in features_files.split(","):
                fp = os.path.join(base_dir, fn)
                assert os.path.exists(fp), 'The feature file:{} is not exits!'.format(fp)
            rows_cl.append(features_files)
    return rows_cl


def load_same_person_utterances(base_dir, meta_file, utterances_num):
    """
    加载同一个人的不同音频片段，来计算错误拒绝率
    :param base_dir: 保持音频mel频谱的文件夹
    :param meta_file: 元文件
    :param utterances_num: 获取一个人的音频判断个数
    :return:
    """
    speakers_utterances_cl = load_data(base_dir, meta_file)

    speaker_id = 0
    while True:
        mel_fps = speakers_utterances_cl[speaker_id % len(speakers_utterances_cl)]
        features_cl = []
        mel_fps_cl = mel_fps.decode("utf-8").split(",")
        mel_fps_samples = random.sample(mel_fps_cl, utterances_num)

        for mel_fp in mel_fps_samples:
            feature = np.load(os.path.join(base_dir, mel_fp)).astype(np.float32)

            clip_index = np.random.randint(low=0, high=len(feature) - hp.partials_n_frames)
            features_cl.append(feature[clip_index:clip_index + hp.partials_n_frames, :])

        yield np.array(features_cl), len(speakers_utterances_cl)

        speaker_id += 1


def get_dataset(mel_feature_base_dir, meta_file):
    """
    laod the training data the generate dataset
    :param mel_feature_base_dir:The directory which contains the features
    :param meta_file: The file contain the meta data
    :return: dataset,num_batch per epoch,shape=[batch,utterances_per_speaker,n_frames,n_mels]
    """

    def _read_py_function(mel_fps):
        features_cl = []
        mel_fps_cl = mel_fps.decode("utf-8").split(",")
        mel_fps_samples = random.sample(mel_fps_cl, pm.utterances_per_speaker)

        for mel_fp in mel_fps_samples:
            feature = np.load(os.path.join(mel_feature_base_dir, mel_fp)).astype(np.float32)

            clip_index = np.random.randint(low=0, high=len(feature) - hp.partials_n_frames)
            features_cl.append(feature[clip_index:clip_index + hp.partials_n_frames, :])

        return np.array(features_cl)

    def _read_caller(mel_fps):
        return tf.py_func(_read_py_function, [mel_fps], tf.float32)

    with tf.device('/cpu:0'):
        rows_cl = load_data(mel_feature_base_dir, meta_file)

        assert len(rows_cl) >= pm.batch_size

        num_batch = len(rows_cl) // pm.batch_size

        dataset = tf.data.Dataset.from_tensor_slices(rows_cl)
        dataset = dataset.shuffle(buffer_size=len(rows_cl), reshuffle_each_iteration=True)

        dataset = dataset.map(_read_caller, num_parallel_calls=pm.thread_num)

        dataset = dataset.repeat()

        dataset = dataset.padded_batch(pm.batch_size, ([None, None, None]))

        dataset = dataset.prefetch(buffer_size=pm.buffer_size * pm.batch_size)

        return dataset, num_batch


def load_one_person_m_utterances(speaker_dir: Path, utterances_num, extension="wav"):
    """
    返回一个speaker的utterances_num条音频，每条音频是经过分割的
    返回格式:
    [
        [bsz1,n_frames,n_mels],
        [bsz2,n_frames,n_mels],
        ...
    ]
    """
    wav_path_cl = [wav_path for wav_path in speaker_dir.glob("*.{}".format(extension))]
    wav_path_cl_samples = random.sample(wav_path_cl, utterances_num)  # 随机抽样utterances_num条音频

    utterances_mels_cl = []
    for fp in wav_path_cl_samples:
        wav, sr = audio.load_wav(str(fp))
        bsz_n_frames_mel = slice_utterance_mel(wav)  # 这里的shape=[bsz,n_frames,n_mels],bsz不是定值
        utterances_mels_cl.append(bsz_n_frames_mel)

    return utterances_mels_cl


def load_multi_person_one_utterances(read_dir: Path, n_speaker, extension="wav"):
    """
    后去n个speaker，每个speaker去随机去一段语音
    :param read_dir: 保持spearker的文件夹
    :param n_speaker: n个speaker
    :return:
    [
        [bsz1,n_frames,n_mels],
        [bsz2,n_frames,n_mels],
        ...
    ]
    """
    speaker_dirs = [speak_dir for speak_dir in read_dir.glob("*")]
    n_speaker_dirs = random.sample(speaker_dirs, n_speaker)

    n_speaker_utterance_mel_cl = []
    for speaker_dir in n_speaker_dirs:
        utterancs_wavs_path = [str(wav_path) for wav_path in Path(str(speaker_dir)).glob("*.{}".format(extension))]
        wav_path = random.sample(utterancs_wavs_path, 1)[0]  # 随机抽样1条音频
        wav, sr = audio.load_wav(wav_path)
        bsz_n_frames_mel = slice_utterance_mel(wav)  # 这里的shape=[bsz,n_frames,n_mels],bsz不是定值
        n_speaker_utterance_mel_cl.append(bsz_n_frames_mel)

    return n_speaker_utterance_mel_cl


def load_multi_person_m_utterances(read_dir: Path, n_speaker, utterances_num):
    """
    返回n个speaker的m条语音的mel频谱
    :return: list,格式为:
    [
        [[bsz1,n_frames,n_mels],[bsz2,n_frames,n_mels],..],
        [[bsz1,n_frames,n_mels],[bsz2,n_frames,n_mels],..],
    ...]
    """
    i = 0
    speaker_dirs = [speak_dir for speak_dir in read_dir.glob("*")]
    speaks_utterrances_cl = []
    while True:
        speak_dir = [i % len(speaker_dirs)]
        speaks_utterrances_cl.append(load_one_person_m_utterances(speak_dir, utterances_num))
        if (i + 1) % n_speaker == 0:
            yield speaks_utterrances_cl, len(speaker_dirs)
            speaks_utterrances_cl.clear()
        i += 1


if __name__ == '__main__':
    pm.batch_size = 8
    pm.test_rate = 0.3
    # pm.meta_file = "test_data/meta.txt"
    # pm.mel_feature_base_dir = "test_data"

    pm.meta_file = "/home/user/tmp/tts/dataset/train-clean-100/LibriSpeech/train-clean-100_features/meta.txt"
    pm.mel_feature_base_dir = "/home/user/tmp/tts/dataset/train-clean-100/LibriSpeech/train-clean-100_features"

    with tf.Session() as sess:
        dataset, num_batch = get_dataset(mel_feature_base_dir=pm.mel_feature_base_dir)

        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(iterator.initializer)

        for _ in range(10):
            features = sess.run(next_element)

            print(np.shape(features))

            features = np.reshape(features, (pm.batch_size * pm.utterances_per_speaker, -1, pm.n_mels))

            print(np.shape(features))
