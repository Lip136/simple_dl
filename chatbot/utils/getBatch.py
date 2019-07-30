# encoding:utf-8

import itertools
import torch

class BatchManager():

    def __init__(self, PAD_token = 0, SOS_token = 1, EOS_token = 2):
        self.PAD_token = PAD_token
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token


    def indexesFromSentence(self, voc, sentence):
        return [voc.word2id[word] for word in sentence.split(' ')] + [self.EOS_token]


    def zeroPadding(self, l):
        return list(itertools.zip_longest(*l, fillvalue=self.PAD_token))

    def binaryMatrix(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self, voc, pair_batch):
        # 按照输入的句子中词的个数排序，大的在前面
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len




