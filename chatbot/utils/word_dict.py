# encoding:utf-8
import unicodedata
import re
PAD_token = 0 # padding
SOS_token = 1 # start
EOS_token = 2 # end


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3


    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于某个计数阈值的词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("keep_words {}/{} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words)/len(self.word2index)
        ))

       # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.word2count = 3

        for word in keep_words:
            self.addWord(word)

MAX_LENGTH = 10
MIN_COUNT = 3
class DataSet:
    def __init__(self, datafile, corpus_name):
        self.corpus_name = corpus_name
        self.datafile = datafile

    def unicodeToAscii(self, s):
        return "".join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normailzeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"[.!?]", r" ", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def readVocs(self):
        print("Reading lines...")

        lines = open(self.datafile, encoding="utf-8").read().strip().split("\n")

        pairs = [[self.normailzeString(s) for s in l.split("\t")] for l in lines]
        voc = Voc(self.corpus_name)
        return voc, pairs

    # 选择小于最大长度的句子
    def filterPair(slef, p):
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def loadPrepareData(self):
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
        return voc, pairs

    # 过滤掉带需要修剪词的对
    def trimRareWords(voc, pairs, MIN_COUNT):
        # Trim words used under the MIN_COUNT from the voc
        voc.trim(MIN_COUNT)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                    len(keep_pairs) / len(pairs)))
        return keep_pairs


