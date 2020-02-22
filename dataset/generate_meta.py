import random
from audio import *
import hparams as hp


def generate_meta_for_test(wav_file: Path, meta_file: Path):
    n = 10
    m = 5

    base_dir = wav_file.parent
    feature_np_fp = base_dir.joinpath(wav_file.name.replace(".wav", ""))

    print("base_dir:", wav_file.parent)
    print("name:", wav_file.name)
    print("feature_np_fp:", feature_np_fp.name)

    wav_mel_feature = wav_to_mel_spectrogram(preprocess_wav(wav_file))
    np.save(file=feature_np_fp, arr=wav_mel_feature)

    feature_files = []
    for i in range(m):
        feature_files.append(feature_np_fp.name + ".npy")

    with open(meta_file, 'w', encoding='utf-8') as f:
        for speak_id in range(n):
            line = "{}|{}\n".format(speak_id, ','.join(feature_files))
            f.write(line)


def generate_meta_for_t(wavs_dir, meta_file):
    n = 20
    m = 5
    wavs_dir = Path(wavs_dir)

    feature_fp_files = []
    for wav_fp in wavs_dir.glob("*.wav"):
        wav_data = preprocess_wav(wav_fp)
        wav_mel_feature = wav_to_mel_spectrogram(wav_data, hp.sampling_rate)
        feature_np_fp = str(wav_fp).replace(".wav", "_{}.npy".format(len(wav_data)))
        np.save(file=feature_np_fp, arr=wav_mel_feature)
        feature_fp_files.append(feature_np_fp)

    feature_files = []
    for i in range(m):
        feature_files.append(Path(feature_fp_files[0]).name)

    with open(meta_file, 'w', encoding='utf-8') as f:
        for speak_id in range(n):
            line = "{}|{}\n".format(speak_id, ','.join(feature_files))
            f.write(line)


if __name__ == '__main__':
    wav_file = "test_data/A2_0.wav"
    meta_file = "test_data/meta.txt"
    # generate_meta_for_test(Path(wav_file), Path(meta_file))

    generate_meta_for_t(wavs_dir="test_data", meta_file=meta_file)
