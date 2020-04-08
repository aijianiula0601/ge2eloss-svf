from pathlib import Path
import random
import os
import sys

prj_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
print('prj_dir:', prj_dir)
sys.path.append(prj_dir)

from eval_test_data.eer_ops import *
import hparams as hp

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

read_dir = '/home/user/tmp/svf/dataset/kefu/test_dataset/test_aishell_embed_vector_v1'

batch_size = 10
utterance_num = 10
min_audio_time = 5  # 限制音频的长度，最小为5s

sess = tf.Session()

embeds_batch = []
eer_cl = []
total_speaker_num = 0
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

        embeds_batch.append(cur_speaker_embed_vectors)

        if len(embeds_batch) >= batch_size:
            embeds = np.array(embeds_batch)
            sm = sess.run(similarity_matrix(tf.convert_to_tensor(embeds)))
            eer = calculate_eer(sm)
            eer_cl.append(eer)

            print('speaker:{} eer:{}'.format(speaker_dir.name, eer))

            total_speaker_num += len(embeds_batch)

            embeds_batch.clear()

print('----------------------------------')
print('batch_size:', batch_size)
print('utterance_num:', utterance_num)
print('speaker个数:', total_speaker_num)
print('min_audio_time:', min_audio_time)
print('avg_eer:', np.mean(np.array(eer_cl)))
print('read_dir:', read_dir)
print('----------------------------------')
