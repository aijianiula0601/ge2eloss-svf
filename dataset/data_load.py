import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import random
import time
import logging

from params import pm
import hparams as hp


def load_data(meta_file):
    """
    load the meta data for training
    The meta data which contains in meta.txt.The lines like: speaker_id|mel_file_paths
    The mel_file_paths is the speark_id's m utterances,which segmented by comma,
        e.g.: The spearker_id is 1 and m=3,which line like: 1|mel1.npy,mel2.npy,mel3.npy
    :param meta_file: The file contain the meta data
    :return: The list contains the mel_file,exclude the speaker_id

    ps：The shape of mel.npy is same,which is [n_frames,n_mels]
    """
    base_dir = str(Path(meta_file).parent)
    rows_cl = []
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.replace("\n", "").split("|")
            features_files = parts[1]

            duration_time = 0

            features_files_ocl = features_files.split(",")

            if len(features_files_ocl) < pm.utterances_per_speaker:  # 只选择包含的音频片段大于训练的音频片段的speaker
                continue
            for fn in features_files_ocl:
                fp = os.path.join(base_dir, fn)
                assert os.path.exists(fp), 'The feature file:{} is not exits!'.format(fp)
                duration_time += int(fp.replace(".npy", "").split("_")[-1])

            if duration_time >= hp.min_second_total_utterances * hp.sampling_rate:  # 只有当前的speaker的音频总时长达到30s以上才进入训练集
                rows_cl.append((base_dir, features_files))
    return rows_cl


def split_train_dev(rows_cl):
    """
    从训练集中划分一部分数据作为验证集,直接把一部分人全部音频分为验证集
    """
    train_row_cl = []
    dev_row_cl = []

    for row in rows_cl:
        if len(dev_row_cl) < pm.batch_size * 3:  # 获取batch_size倍数的speaker训练
            dev_row_cl.append(row)
            continue

        train_row_cl.append(row)

    dev_meta_file = "/tmp/svf_dev_meta_{}.txt".format(int(time.time()))

    with open(dev_meta_file, 'w', encoding='utf-8') as f:
        for row in dev_row_cl:
            f.write("{}\t{}\n".format(row[0], row[1]))

    logging.info("验证集个数:{} 训练集个数:{}".format(len(dev_row_cl), len(train_row_cl)))
    logging.info("验证集数据保存至:{}".format(dev_meta_file))

    return train_row_cl, dev_row_cl


def get_dev_dataset_for_test(dev_meta_file):
    """
    加载验证集，做测试
    """
    row_cl = []
    with open(dev_meta_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            part = line.replace("\n", "").split("\t")

            row_cl.append((part[0], part[1]))

    logging.info("加载speaker个数:{}".format(len(row_cl)))

    dataset, num_batch = get_dataset_iter(row_cl)

    return dataset, num_batch


def get_dataset_iter(rows_cl):
    def _read_py_function(row):
        mel_feature_base_dir, mel_fps = row[0].decode("utf-8"), row[1].decode("utf-8")

        features_cl = []
        mel_fps_cl = mel_fps.split(",")
        mel_fps_samples = random.sample(mel_fps_cl, pm.utterances_per_speaker)
        for mel_fp in mel_fps_samples:
            feature = np.load(os.path.join(mel_feature_base_dir, mel_fp)).astype(np.float32)
            clip_index = np.random.randint(low=0, high=len(feature) - hp.partials_n_frames)
            features_cl.append(feature[clip_index:clip_index + hp.partials_n_frames, :])

        return np.array(features_cl)

    def _read_caller(row):
        return tf.py_func(_read_py_function, [row], tf.float32)

    num_batch = len(rows_cl) // pm.batch_size

    dataset = tf.data.Dataset.from_tensor_slices(rows_cl)
    dataset = dataset.shuffle(buffer_size=len(rows_cl), reshuffle_each_iteration=True)

    dataset = dataset.map(_read_caller, num_parallel_calls=pm.thread_num)

    dataset = dataset.repeat()

    dataset = dataset.padded_batch(pm.batch_size, ([None, None, None]))

    # dataset = dataset.prefetch(buffer_size=pm.buffer_size * pm.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_batch


def get_dataset(meta_files):
    """
    laod the training data the generate dataset
    :param meta_files: [list] contain the meta files,the meta_file must be the same directory with dataset
    :return: dataset,num_batch per epoch,shape=[batch,utterances_per_speaker,n_frames,n_mels]
    """

    with tf.device('/cpu:0'):
        rows_cl = []
        for meta_file in meta_files:
            rows_cl += load_data(meta_file)

        random.shuffle(rows_cl)

        assert len(rows_cl) >= pm.batch_size

        train_row_cl, dev_row_cl = split_train_dev(rows_cl)

        train_dataset, train_num_batch = get_dataset_iter(train_row_cl)
        dev_dataset, dev_num_batch = get_dataset_iter(dev_row_cl)

        return train_dataset, train_num_batch, dev_dataset, dev_num_batch


if __name__ == '__main__':
    pm.batch_size = 8
    pm.test_rate = 0.3
    pm.meta_file = "/Users/jia/IdeaProjects/github/ge2eloss-svf/dataset/test_data/meta.txt"

    with tf.Session() as sess:
        train_dataset, train_num_batch, dev_dataset, dev_num_batch = get_dataset(meta_files=[pm.meta_file])

        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(iterator.initializer)

        for _ in range(5):
            features = sess.run(next_element)

            print(np.shape(features))

            # features = np.reshape(features, (pm.batch_size * pm.utterances_per_speaker, -1, pm.n_mels))
            # print(np.shape(features))
