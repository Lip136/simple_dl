# -*- coding:utf8 -*-
"""
This module implements data process strategies.
"""

import logging
import numpy as np
import jieba
import itertools
import torch
from sklearn.model_selection import train_test_split

class ChatDataset(object):
    """
    This module implements the APIs for loading and using dataset
    """
    def __init__(self, data_path=None, MAX_LENGTH=20):
        self.logger = logging.getLogger("chatbot")

        self.MAX_LENGTH = MAX_LENGTH

        if data_path:
            self.data_set = self._load_dataset(data_path)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))


    def _load_dataset(self, data_path):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        def _seg(sentence):
            return list(jieba.cut(sentence))
        import re
        with open(data_path) as fin:

            for line in fin:
                sample = dict()
                # 中文 - 这里是列表
                pair = [s for s in line.strip('\n').split("\t")]
                sample["question"] = _seg(pair[0])
                sample["answer"] = _seg(pair[1])
                # 英文
                # pair = re.sub("[,.?]", "", line).split("\t")

                if not self.filterPair(pair):
                    continue

                # sample["question"] = pair[0]
                # sample["answer"] = pair[1]
                data_set.append(sample)

        self.train_set, self.dev_set = train_test_split(data_set, test_size=0.2, random_state=42)

        return data_set

    # 选择小于最大长度的句子
    def filterPair(self, pair):
        return len(pair[0]) < self.MAX_LENGTH and len(pair[1]) < self.MAX_LENGTH


    # 将句子中的词变成id，最后再加个结束符
    def indexesFromSentence(self, voc, sentence):
        return [voc.get_id(word) for word in sentence] + [voc.get_id("<end>")]

    # 将shape变成 batch_size * seq_length => seq_length * batch_size
    def zeroPadding(self, l, PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=PAD_token))


    def binaryMatrix(self, l, PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch, voc.get_id("<pad>"))
        padVar = torch.tensor(padList)
        return padVar, lengths

    # 输出加上mask: pad为0, 词语为1
    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch, voc.get_id("<pad>"))
        padVar = torch.tensor(padList)  # LongTensor

        # mask = self.binaryMatrix(padList, voc.get_id("<pad>"))
        # mask = torch.tensor(mask) # ByteTensor
        mask = (padVar != voc.get_id("<pad>"))
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def _one_mini_batch(self, voc, pair_batch):
        # 按照输入的句子中词的个数排序，大的在前面
        pair_batch.sort(key=lambda x: len(x["question"]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair["question"])
            output_batch.append(pair["answer"])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len


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
            data_size = len(self.train_set)
            # indices = np.arange(data_size)
            # if shuffle:
            #     np.random.shuffle(indices)
            for batch_start in np.arange(0, data_size, batch_size):
                batch_data = self.train_set[batch_start: batch_start + batch_size]
                # batch_data = [data[i] for i in batch_indices]
                yield self._one_mini_batch(voc, batch_data)
        else:
            data_size = len(self.dev_set)
            for batch_start in np.arange(0, data_size, batch_size):
                batch_data = self.dev_set[batch_start: batch_start + batch_size]
                yield self._one_mini_batch(voc, batch_data)
