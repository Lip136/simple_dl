# encoding:utf-8
import os
import itertools
import torch
import word_dict
import random

PAD_token = 0 # padding
SOS_token = 1 # start
EOS_token = 2 # end

class HandleData:

    def __init__(self):
        pass

    def indexesFromSentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=PAD_token):
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
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len

    def getBatchData(self):
        MAX_LENGTH = 10
        corpus_name = "cornell movie-dialogs corpus"
        corpus = os.path.join("F:\\code\\py3.6\\simple_dl\\chatbot\\data", corpus_name)
        datafile = os.path.join(corpus, "formatted_movie_lines.txt")
        save_dir = os.path.join("F:\\code\\py3.6\\simple_dl\\chatbot\\data", "model")
        voc, pairs = word_dict.DataSet(datafile, corpus_name).loadPrepareData()


        # Example for validation
        small_batch_size = 5
        batches = self.batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
        input_variable, lengths, target_variable, mask, max_target_len = batches

        return voc, pairs, input_variable, lengths, target_variable, mask, max_target_len
        # print("input_variable:", input_variable)
        # print("lengths:", lengths)
        # print("target_variable:", target_variable)
        # print("mask:", mask)
        # print("max_target_len:", max_target_len)

