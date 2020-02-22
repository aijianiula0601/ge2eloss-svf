## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10  # In milliseconds
mel_n_channels = 40

## Audio
sampling_rate = 8000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160  # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80  # 800 ms

## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6

## Audio volume normalization
audio_norm_target_dBFS = -30

ativate_detetion_dB = 20

int16_max = (2 ** 15) - 1

tisv_frames = 180

# ----------------------
# 训练模型超参数
# ----------------------
## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 4

# ----------------------
# 限制训练集超参数
# ----------------------
min_second_total_utterances = 30  # 训练数据的speaker必须所有音频达到30s以上
num_dev_batch = 3
