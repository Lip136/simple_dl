# encoding:utf-8
"""
功能:
数据管理
"""

import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import glob, os
import itertools
from utils import load_dataset, load_and_trim

class ASRDataset(object):
    """
    This module implements the APIs for loading and using dataset
    """
    def __init__(self, data_path=None, MAX_LENGTH=20):
        self.logger = logging.getLogger("asr")

        self.MAX_LENGTH = MAX_LENGTH

        if data_path:
            # self.train_set, self.dev_set = self._load_dataset(data_path)
            self.train_set = self._load_dataset(data_path, "train")
            self.dev_set = self._load_dataset(data_path, "dev")
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        else:
            self.train_set = load_dataset("train")
            self.dev_set = load_dataset("dev")
            print('Train set size: {} audio files.'.format(len(self.train_set)))

    def _load_dataset(self, data_path, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        path_set = glob.glob(os.path.join(data_path, '%s_data/*.npy'%mode))
        path_set.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))

        with open("aidataaudio_%s.txt"%mode, "r") as f:
            lable_set = f.readlines()

        assert len(lable_set) == len(path_set)
        for i in range(len(lable_set)):
            data_set.append([path_set[i], list(lable_set[i].strip())])


        # for path in path_set:
        #     sample = [path.rstrip(".trn")]
        #     with open(path, "r") as f:
        #         sample.append(list("".join(f.readline().strip().split())))
        #     data_set.append(sample)

        # train_set, dev_set = train_test_split(data_set, test_size=0.2, random_state=42)
        # return train_set, dev_set
        return data_set

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
        # 这可以检测一遍数据
        # if np.any(np.isnan(np.array(padList, dtype=np.float32))):
        #     print("输入数据有问题")
        #     np.save("input_data_nan.npy", np.array(padList, dtype=np.float32))

        padVar = torch.tensor(padList)
        return padVar, lengths

    # 输出加上mask: pad为0, 词语为1
    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):

        indexes_batch = [voc.convert_to_ids(sentence) for sentence in l]
        target_len = torch.tensor([len(indexes) for indexes in indexes_batch])
        max_target_len = max(target_len).item()
        padList = self.zeroPadding(indexes_batch, voc.get_id("_"))
        padVar = torch.tensor(padList).t()  # LongTensor

        return padVar, target_len, max_target_len

    # Returns all items for a given batch of pairs
    def _one_mini_batch(self, voc, pair_batch):

        # pair_batch.sort(key=lambda x: len(x[1]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            # audio = load_and_trim(pair[0], mfcc_dim=39)
            audio = np.load(pair[0])
            input_batch.append(audio)
            output_batch.append(pair[1])
        inp, inp_len = self.inputVar(input_batch, voc)
        # print(inp_len.shape)
        output, target_len, max_target_len = self.outputVar(output_batch, voc)
        return inp, inp_len, output, target_len, max_target_len


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


if __name__ == "__main__":
    dataset = ASRDataset("./")
