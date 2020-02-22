from pathlib import Path
import random
import os
import sys
import numpy as np

"""
测试错误拒绝率
"""

prj_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
print('----prj_dir:', prj_dir)
sys.path.append(prj_dir)

from eval_test import frp_fap
import hparams as hp

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

read_dir = '/home/huangjiahong/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16_embed'

utterance_num = 15
frp_cl = []
threshold_p = 0.656

min_audio_time = 5  # 限制音频的长度，最小为5s

for speaker_dir in Path(read_dir).glob("*"):

    cur_speaker_feature_files = []

    for fp in speaker_dir.glob("*.npy"):

        duration_time = int(fp.name.replace(".npy", "").split("_")[-1])

        if duration_time >= min_audio_time * hp.sampling_rate:
            cur_speaker_feature_files.append(fp)

    if len(cur_speaker_feature_files) >= utterance_num:
        mel_fps_samples = random.sample(cur_speaker_feature_files, utterance_num)

        cur_speaker_embed_vectors = []
        for fp in mel_fps_samples:
            embed_vector = np.load(fp).astype(np.float32)
            cur_speaker_embed_vectors.append(embed_vector)

        speaker_utterances_embeds = np.array(cur_speaker_embed_vectors)
        # 计算错误拒绝率
        intra_class_matrix = np.matmul(speaker_utterances_embeds, speaker_utterances_embeds.T)

        frp = frp_fap.frp(intra_class_matrix, threshold_p=threshold_p)

        print("speaker:{} frp:{}".format(speaker_dir.name, frp))
        # print('samples:', mel_fps_samples)
        frp_cl.append(frp)

print('阀值:{} avg_frp:{}'.format(threshold_p, np.mean(np.array(frp_cl))))
print('read_dir:', read_dir)
