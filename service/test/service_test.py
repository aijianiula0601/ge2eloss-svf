# -*- coding: utf-8 -*-

import os
import sys
import requests
from timeit import default_timer as timer
import json
import scipy.io.wavfile as wav
from urllib import parse

import audio as audio_ops


def audio_body(audio_file):
    wav, sr = audio_ops.load_wav(audio_file)

    body_l = (wav.tolist(), sr)

    return body_l


def get_result(audio_file):
    url = 'http://localhost:5112/api/1.0/asr?%s'

    params = {'requestId': '2700987724946222121'}
    # params = {'ifUrl': 1}
    url = url % parse.urlencode(params)

    body = audio_body(audio_file)

    bodys = {1: body, 2: body, 3: body}

    audios_bodys = {'wav_bodys': bodys}

    # post数据过去
    resp = requests.post(url, data=json.dumps(audios_bodys))

    return json.loads(resp.text)


if __name__ == '__main__':
    audio_file = 'E:/Download/A2_0.wav'

    # for i in range(50):
    process_start = timer()

    result = get_result(audio_file)

    print(result)

    # 耗时
    process_time = timer() - process_start

    print('process time {:.5}ms:'.format(process_time * 1000))
