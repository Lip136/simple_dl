# encoding:utf-8
import numpy as np

class embed_word(object):
    '''
    功能： 构建{词：向量}一一映射
    输入： 读入的文本列表:["word", "dict"]
    输出： word2n_dict, number2w_dict
    '''
    def __init__(self, sentences):
        self.sentences = sentences
        self.text = list(set(" ".join(sentences).split()))
        self.n_class = len(self.text)
        self.word2n = {}
        self.number2w = {}
        self.input_batch = []
        self.target_batch = []

    def word_dict(self):
        n_embed = np.eye(self.n_class) # one-hot 表示
        self.word2n = {w: n_embed[n].tolist() for n, w in enumerate(self.text)}
        self.number2w = {tuple(n_embed[n].tolist()): w for n, w in enumerate(self.text)}

        return self.word2n, self.number2w, self.n_class

    def get_batch(self):
        for sen in self.sentences:
            word = sen.split()
            input_data = [self.word2n[n] for n in word[:-1]]  # The first two words
            target_data = self.word2n[word[-1]]
            self.input_batch.append(np.array(input_data))  # one-hot 表示
            self.target_batch.append(np.array(target_data))

        return self.input_batch, self.target_batch

if __name__ == "__main__":
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    embed = embed_word(sentences)
    word2n, number2w, n_class = embed.word_dict()
    input_batch, target_batch = embed.get_batch()

    print(input_batch, '\n', target_batch)