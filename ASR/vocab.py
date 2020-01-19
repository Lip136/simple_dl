# encoding:utf-8
"""
功能:
1.获取id
2.获取token
3.获取size
4.句子映射
"""
import os
import pickle
import utils

class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, mfcc_dim, filenames=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower



        self.PAD_token = "<pad>"
        self.SOS_token = "<start>"
        self.EOS_token = "<end>"
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        # self.initial_tokens.extend([self.PAD_token, self.SOS_token, self.EOS_token, self.unk_token])

        self.initial_tokens.extend([self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filenames is not None:
            for filename in filenames:
                self.load_from_file(filename)
            # 明天来把这改了
            self.audio_mean, self.audio_std = utils.get_standard_params(filenames, mfcc_dim)



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

        with open(file_path, 'r') as f:
            sentence = f.readline()
            tokens = "".join(sentence.strip().split())
            for token in tokens:
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


if __name__ == "__main__":
    # import glob
    # filenames = glob.glob(os.path.join("data", "*.trn"))
    # root = "/home/hg/data/aidatatang_200zh/corpus/train"
    root = "/home/user/nlp/asr/aidatatang_200zh/corpus/train"
    filenames = []
    for dirpath, dirfiles, filename in os.walk(root):
        for name in filename:
            if os.path.splitext(name)[-1] == ".trn":
                filenames.append(os.path.join(dirpath, name))

    vocab = Vocab(filenames, initial_tokens=["_"])
    print(vocab.size())
    print(vocab.audio_mean, vocab.audio_std)
    # with open("aidatatang-200.vocab", "wb") as f:
    #     pickle.dump(vocab, f)

