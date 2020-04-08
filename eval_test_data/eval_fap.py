from pathlib import Path
import random
import os
import sys
import numpy as np

"""
测试错误接受率
"""

prj_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
print('----prj_dir:', prj_dir)
sys.path.append(prj_dir)

from eval_test_data.utils import frp_fap
import hparams as hp

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

read_dir = '/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16_embed'

speaker_num = 20

fap_cl = []
threshold_p = 0.656

speakers_utterance = []

min_audio_time = 5  # 限制音频的长度，最小为5s

for speaker_dir in Path(read_dir).glob("*"):
    cur_speaker_feature_files = []

    for fp in speaker_dir.glob("*.npy"):

        duration_time = int(fp.name.replace(".npy", "").split("_")[-1])

        if duration_time >= min_audio_time * hp.sampling_rate:
            cur_speaker_feature_files.append(fp)

    if len(cur_speaker_feature_files) <= 1:
        continue

    mel_fps_samples = random.sample(cur_speaker_feature_files, 1)  # 从一个speaker中抽样一条音频出来

    fp = mel_fps_samples[0]
    embed_vector = np.load(fp).astype(np.float32)

    speakers_utterance.append(embed_vector)

    if len(speakers_utterance) >= speaker_num:
        batch_speakers_utterance = np.array(speakers_utterance)

        # 计算错误拒绝率
        intra_class_matrix = np.matmul(batch_speakers_utterance, batch_speakers_utterance.T)

        fap = frp_fap.fap(intra_class_matrix, threshold_p=threshold_p)

        print("speaker:{} fap:{}".format(speaker_dir.name, fap))
        fap_cl.append(fap)

        speakers_utterance.clear()

print('阀值:{} avg_fap:{}'.format(threshold_p, np.mean(np.array(fap_cl))))
print('read_dir:', read_dir)
