#encoding=utf-8
'''
功能:文件操作
解释：FileObj -> 打开文件，返回list
     Voc     -> 给sentence，返回word2id(没有标识符), id2word, word2count(没有标识符), num_words
     DataSet -> 返回voc, pairs
'''
import codecs
class FileObj(object):
    def __init__(self, filepath):
        self.filepath = filepath
    # 按行读入数据，返回一个List
    def read_lines(self):

        with codecs.open(self.filepath,'r','utf-8') as file_obj:
            lines = file_obj.readlines()
            self.sentences = [line.strip('\r\n') for line in lines]

        return self.sentences

class Voc(object):

    def __init__(self, name, PAD_token = 0, SOS_token = 1, EOS_token = 2, trimmed=False):
        self.name = name
        self.PAD_token = PAD_token # padding
        self.SOS_token = SOS_token # start
        self.EOS_token = EOS_token # end
        self.trimmed = trimmed

        self.word2id = {}
        self.id2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.word2count = {}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.num_words
            self.id2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于某个计数阈值的词
    def trim(self, min_count):
        if not self.trimmed: return None

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("keep_words {}/{} = {:.4f}".format(
            len(keep_words), len(self.word2id), len(keep_words)/len(self.word2id)
        ))

       # Reinitialize dictionaries
        self.word2id = {}
        self.id2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.word2count = {}
        self.word2count = 3

        for word in keep_words:
            self.addWord(word)

import re
import unicodedata
import dill as pickle
import jieba

# 中文分词，先采用jiaba
class Seg(object):
    def __init__(self, stopword_filepath="../data/stopword.txt", stopword=False):
        self.stopword = stopword
        if stopword:
            self.stopwords = FileObj(stopword_filepath).read_lines()
        else:
            self.stopwords = []

    def cut(self, sentence):
        seg_list = jieba.cut(sentence)

        results = []
        for seg in seg_list:
            if seg in self.stopwords and self.stopword:
                continue
            results.append(seg)
        return results

    def cut_for_search(self, sentence):
        seg_list = jieba.cut_for_search(sentence)

        results = []
        for seg in seg_list:
            if seg in self.stopwords and self.stopword:
                continue
            results.append(seg)
        return results

class DataSet():
    def __init__(self, datafile, corpus_name, MAX_LENGTH = 60, MIN_COUNT = 3):
        self.corpus_name = corpus_name
        self.datafile = datafile
        self.MAX_LENGTH = MAX_LENGTH
        self.MIN_COUNT = MIN_COUNT

    # 英文专用
    def unicodeToAscii(self, s):
        return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"[.!?]", r" ", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    # 中文分词
    def seg_sentence(self, s):

        return " ".join(Seg().cut(s))

    def readVocs(self):
        print("Reading lines...")

        lines = codecs.open(self.datafile,'r','utf-8').readlines()

        # 这里给的格式每一行是：-输入句子\t目标句子
        pairs = [[self.seg_sentence(s).strip() for s in l.strip('\r\n').split("\t")] for l in lines]
        voc = Voc(self.corpus_name)
        return voc, pairs

    # 选择小于最大长度的句子
    def filterPair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and len(p[1].split(' ')) < self.MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def prepareData(self):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs()
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words ...")

        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counting words:", voc.num_words)
        # return voc, pairs
        self.saveFile(voc, pairs)

    def saveFile(self, voc, pairs):

        with open("../data/vocData_{}.pkl".format(self.corpus_name), 'wb') as outp:
            pickle.dump(voc, outp)
            pickle.dump(pairs, outp)
        print("Save vocData_{}.pkl".format(self.corpus_name))

    def loadData(self, voc_path):

        with open("{}/vocData_{}.pkl".format(voc_path, self.corpus_name), 'rb') as inp:
            voc = pickle.load(inp)
            pairs = pickle.load(inp)

        return voc, pairs

import os
if __name__ == "__main__":
    corpus_name = "qingyun"
    filePath = "/media/ptface02/H1/dataSet/中文聊天语料/chaotbot_corpus_Chinese/clean_chat_corpus"
    datafile = os.path.join(filePath, "{}.tsv".format(corpus_name))

    dataSet = DataSet(datafile, corpus_name)
    # dataSet.prepareData()
    voc, pairs = dataSet.loadData("../data")
    print(len(voc.word2id), len(voc.id2word), voc.num_words, len(voc.word2count))
    print(pairs[:2])


