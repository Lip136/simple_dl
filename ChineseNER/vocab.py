#encoding=utf-8

import numpy as np
import re
import torch
from torch import nn

class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.id2tag = {}
        self.tag2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.PAD_token = "<pad>"
        self.SOS_token = "<start>"
        self.EOS_token = "<end>"
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.PAD_token, self.SOS_token, self.EOS_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def load_from_file(self, file_path):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """
        import jieba
        for line in open(file_path, 'r'):
            sentence = []
            sentence.append()
            sentence = re.sub("[ \n,.]", "", line)
            for token in tokens:
                if token == "\t":
                    continue
                else:
                    self.add(token)

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = torch.randn(self.size(), self.embed_dim)
        nn.init.xavier_normal_(self.embeddings)  # math.sqrt(6 / (self.size() * self.embed_dim))
        for token in [self.PAD_token, self.SOS_token, self.EOS_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = torch.zeros([self.embed_dim])


    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            line_num = 0
            for line in fin:
                line_num += 1
                contents = line.strip().split()
                token = contents[0]
                if token not in self.token2id:
                    continue
                try:
                    trained_embeddings[token] = list(map(np.float32, contents[1:]))
                except:
                    print("词向量文件中 {}-{} 有点问题".format(line_num, token))

                    continue
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        # filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        # self.token2id = {}
        # self.id2token = {}
        # for token in self.initial_tokens:
        #     self.add(token, cnt=0)
        # for token in filtered_tokens:
        #     self.add(token, cnt=0)
        # load embeddings
        self.embeddings = torch.randn([self.size(), self.embed_dim])
        nn.init.xavier_normal_(self.embeddings)
        no_vec = 0

        for token in self.token2id.keys():
            # 初始的tokens词向量都为0
            if token in self.initial_tokens:
                self.embeddings[self.get_id(token)] = torch.zeros([1, self.embed_dim])
                # torch.tensor(np.zeros(self.embed_dim))
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = torch.tensor(trained_embeddings[token])
            else:
                no_vec += 1
                continue

        print("有{}个不在词向量文件中".format(no_vec))


    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens