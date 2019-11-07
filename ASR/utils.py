# encoding:utf-8
"""
功能:
1.获取mean和std
2.获取token
3.获取size
4.句子映射
"""

import numpy as np
import re, os
import torch
from torch import nn
from tqdm import tqdm
import pickle
import glob, random
from dataset import ASRDataset
# random.seed(42)

def standardDate(inp_audio, vocab):
    # bs * 13 * seq_len
    # mfcc_mean = [ -4.13904153,   5.08867636, -26.67533752,  15.00214156,
    #               -22.40365664,  -0.81912315, -17.1327757 ,   6.57387586,
    #               -19.08609289,   5.0187884 , -10.69399481,  -1.66688138,
    #               -2.02195375]
    # mfcc_std = [ 3.00484029, 14.55029426, 14.36740557, 19.94853475, 16.16290146,
    #              22.27750743, 18.08026712, 17.60262745, 16.08046824, 15.55921372,
    #              13.48638884, 12.48479331, 10.5886443 ]
    mfcc_mean, mfcc_std = vocab.audio_mean, vocab.audio_std

    mfcc_mean = torch.from_numpy(mfcc_mean).unsqueeze(dim=0).unsqueeze(dim=2)
    mfcc_std = torch.from_numpy(mfcc_std).unsqueeze(dim=0).unsqueeze(dim=2)

    return (inp_audio - mfcc_mean)/(mfcc_std + 1e-14)

def get_standard_params(filenames, sample_ratio=0.1):
    audio_paths = filenames
    data_manager = ASRDataset()
    audio_paths = random.sample(audio_paths, int(len(audio_paths)*sample_ratio))
    means, stds = [], []

    for audio_path in tqdm(audio_paths):
        a_mfcc = data_manager.load_and_trim(audio_path.rstrip(".trn"))

        means.append(a_mfcc.mean(axis=1))
        stds.append(a_mfcc.std(axis=1))

    audio_mean = np.stack(means, axis=0).mean(axis=0)
    audio_std = np.stack(stds, axis=0).std(axis=0)
    return audio_mean, audio_std

if __name__ == "__main__":
    print(get_standard_params("data"))

