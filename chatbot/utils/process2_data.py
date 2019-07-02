# encoding:utf-8
import unicodedata
import re
import word_dict
import os


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normailzeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[.!?]", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def readVocs(datafile, corpus_name):
    print("Reading lines...")

    lines = open(datafile, encoding="utf-8").read().strip().split("\n")

    pairs = [[normailzeString(s) for s in l.split("\t")] for l in lines]
    voc = word_dict.Voc(corpus_name)
    return voc, pairs

# 选择小于最大长度的句子
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
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

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

MAX_LENGTH = 10
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("../data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
save_dir = os.path.join("../data", "model")
voc, pairs = loadPrepareData(corpus_name, datafile)

MIN_COUNT = 3    # Minimum word count threshold for trimming

pairs = trimRareWords(voc, pairs, MIN_COUNT)

