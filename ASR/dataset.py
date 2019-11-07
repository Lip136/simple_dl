# encoding:utf-8
"""
功能:
数据管理
"""

import logging
import numpy as np
import jieba
import torch
from sklearn.model_selection import train_test_split
import glob, os
import librosa
from tqdm import tqdm
import random, pickle, itertools

class ASRDataset(object):
    """
    This module implements the APIs for loading and using dataset
    """
    def __init__(self, data_path=None, MAX_LENGTH=20):
        self.logger = logging.getLogger("chatbot")

        self.MAX_LENGTH = MAX_LENGTH

        if data_path:
            self.train_set, self.dev_set = self._load_dataset(data_path)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))


    def _load_dataset(self, data_path):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        path_set = glob.glob(os.path.join(data_path, '*.trn'))

        for path in path_set:
            sample = [path.rstrip(".trn")]

            with open(path, "r") as f:
                sample.append(list("".join(f.readline().strip().split())))

            data_set.append(sample)

        train_set, dev_set = train_test_split(data_set, test_size=0.2, random_state=42)

        return train_set, dev_set


    def load_and_trim(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=None) # 速度慢主要在这儿
        energy = librosa.feature.rms(audio)
        frames = np.nonzero(energy >= np.max(energy)/5)
        indices = librosa.core.frames_to_samples(frames)[1]
        audio_trim = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

        audio = self.extract_mfcc(audio_trim, sr)
        return audio

    def extract_mfcc(self, audio, sr):

        a_mfcc = librosa.feature.mfcc(audio, sr, n_mfcc=13)

        return a_mfcc

    # 将句子中的词变成id，最后再加个结束符
    def indexesFromSentence(self, voc, sentence):
        return [voc.get_id(word) for word in sentence]

    # 将shape变成 batch_size * seq_length => seq_length * batch_size
    def zeroPadding(self, l, PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=PAD_token))


    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        """
        param:
        l is list
        len(l) = batch_size
        l[0].shape = (n_mfcc, seq_length)
        """

        lengths = torch.tensor([sample.shape[1] for sample in l])
        max_length = max(lengths).item()
        padded = lambda sample : np.pad(sample, ((0, 0), (0, max_length - sample.shape[1])), constant_values=0.0)

        padList = [padded(sample) for sample in l]

        padVar = torch.tensor(padList)
        return padVar, lengths

    # 输出加上mask: pad为0, 词语为1
    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        target_len = torch.tensor([len(indexes) for indexes in indexes_batch])
        max_target_len = max(target_len).item()
        padList = self.zeroPadding(indexes_batch, voc.get_id("_"))
        padVar = torch.tensor(padList).t()  # LongTensor

        return padVar, target_len

    # Returns all items for a given batch of pairs
    def _one_mini_batch(self, voc, pair_batch):

        pair_batch.sort(key=lambda x: len(x[1]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            audio = self.load_and_trim(pair[0])
            input_batch.append(audio)
            output_batch.append(pair[1])
        inp, inp_len = self.inputVar(input_batch, voc)
        output, target_len = self.outputVar(output_batch, voc)
        return inp, inp_len, output, target_len


    def gen_mini_batches(self, mode, batch_size, voc):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if mode == "train":
            data_set = self.train_set
        else:
            data_set = self.dev_set

        data_size = len(data_set)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_data = data_set[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(voc, batch_data)


