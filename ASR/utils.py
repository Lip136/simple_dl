# encoding:utf-8
"""
功能:
1.获取mean和std
2.获取token
3.获取size
4.句子映射
"""

import numpy as np
import torch
from tqdm import tqdm
import random, os, librosa
from multiprocessing import Process

# random.seed(42)

def standardDate(inp_audio, vocab):
    mfcc_mean, mfcc_std = vocab.audio_mean, vocab.audio_std

    mfcc_mean = torch.from_numpy(mfcc_mean).unsqueeze(dim=0).unsqueeze(dim=2)
    mfcc_std = torch.from_numpy(mfcc_std).unsqueeze(dim=0).unsqueeze(dim=2)

    return (inp_audio - mfcc_mean)/(mfcc_std + 1e-14)

def get_standard_params(filenames, mfcc_dim, sample_ratio=0.7):
    audio_paths = filenames
    audio_paths = random.sample(audio_paths, int(len(audio_paths)*sample_ratio))

    means, stds = [], []

    # 多进程

    for audio_path in tqdm(audio_paths):
        a_mfcc = load_and_trim(audio_path.rstrip(".trn") + ".wav", mfcc_dim)

        means.append(a_mfcc.mean(axis=1))
        stds.append(a_mfcc.std(axis=1))

    audio_mean = np.stack(means, axis=0).mean(axis=0)
    audio_std = np.stack(stds, axis=0).std(axis=0)
    return audio_mean, audio_std


def load_dataset(data_path):
    """
    Loads the dataset
    Args:
        data_path: the data file to load
    """
    # root = "/home/hg/data/aidatatang_200zh/corpus/" + data_path
    root = "/home/user/nlp/asr/aidatatang_200zh/corpus/" + data_path
    filenames = []
    for dirpath, dirfiles, filename in os.walk(root):
        for name in filename:
            if os.path.splitext(name)[-1] == ".trn":
                filenames.append(os.path.join(dirpath, name))
    data_set = []

    for path in filenames:
        # sample = [audio path, a list of words]
        sample = [path.rstrip(".trn") + ".wav"]
        with open(path, "r") as f:
            sample.append(list("".join(f.readline().strip().split())))

        data_set.append(sample)

    return data_set

def load_and_trim(audio_path, mfcc_dim=13):
    audio, sr = librosa.load(audio_path, sr=None) # 速度慢主要在这儿
    # energy = librosa.feature.rms(audio)
    # frames = np.nonzero(energy >= np.max(energy)/5)
    # indices = librosa.core.frames_to_samples(frames)[1]
    # audio_trim = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    # mfcc feature

    audio = librosa.feature.mfcc(audio, sr, n_mfcc=mfcc_dim)
    return audio



if __name__ == "__main__":
    # print(get_standard_params("data"))
    # audio = load_and_trim("data/A11_0.wav", mfcc_dim=39)
    # print(audio.shape[1])
    import glob
    paths = glob.glob("THCHS-30_data/*.trn")
    THCHS = open("THCHS-30.txt", "w")
    for i in tqdm(range(len(paths))):
        tmp = load_and_trim(paths[i].rstrip(".trn"), mfcc_dim=39)
        np.save("THCHS_data/data_set_%s.npy"%i, tmp)
        with open(paths[i], "r") as f:
            THCHS.write("".join(f.readline().strip().split()) + "\n")
    THCHS.close()


