import random
import os
import sys
import shutil
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))))
print('prj_dir:', prj_dir)
sys.path.append(prj_dir)

import audio
import hparams as hp

"""
客服数据抽样12条语音，用户测试
"""

read_dir = '/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16'
save_dir = '/home/user/tmp/svf/dataset/kefu/record_kefu_speakers_segment_16_sample'

min_second = 3
sample_num = 10


def sample_speaker_wav(speaker_dir, save_dir):
    speaker_dir_wav_cl = [fp for fp in speaker_dir.glob("*.wav")]

    random.shuffle(speaker_dir_wav_cl)

    i = 0
    for fp in speaker_dir_wav_cl:
        wav = audio.preprocess_wav(fp, hp.sampling_rate)

        if i < sample_num and len(wav) >= min_second * hp.sampling_rate:
            save_speaker_dir = Path(save_dir).joinpath(speaker_dir.name)
            save_speaker_dir.mkdir(exist_ok=True)

            shutil.copy(str(fp), str(save_speaker_dir))

            i += 1
    return 1


if __name__ == '__main__':


    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=8)
    all_task = [executor.submit(
        partial(sample_speaker_wav, speaker_dir, save_dir)) for speaker_dir in
        Path(read_dir).glob("*")]

    speaker_n = 0
    for task in tqdm(all_task):
        n = task.result()
        speaker_n += n

    print("共处理 speaker个数:{}".format(speaker_n))
