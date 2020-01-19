# encoding=utf-8

import pickle, torch
from vocab import Vocab
from utils import standardDate, load_and_trim, load_dataset
from decoder import greedy, beam_search
from layers.layer_v9 import ASR_lstm
# v5 0.78 0.19
# v4 0.92 0.25

from tqdm import tqdm

# load model and vocab
vocab = pickle.load(open("model/aidatatang-200.vocab", "rb"))
checkpoint = torch.load("model/asraidatatang-200_v9.bin",  map_location="cpu")
model = ASR_lstm(39, 512, vocab.size(), "cpu", num_layers=2)
# model = nn.DataParallel(model)
model.load_state_dict(checkpoint["model"])
# model = model.module
print(checkpoint["min_loss"])
from visdom import Visdom
# viz = Visdom(env="asr", server='http://10.3.27.36', port=8097)
# assert viz.check_connection()


# audio_path = ["data/T0055G0036S0002.wav", ["机", "票", "广", "州", "济", "南"]]
def audio2text(audio_data):
    # 打印标签
    label = "".join(audio_data[1])

    # 预处理数据
    # a_mfcc = load_and_trim(audio_data[0], mfcc_dim=39)
    a_mfcc = np.load(audio_data[0])
    X = torch.from_numpy(a_mfcc).unsqueeze(dim=0)
    X = standardDate(X, vocab)
    # 预测
    model.eval()
    y_pred = model(X)
    y_pred = y_pred.squeeze()
    # print(y_pred.shape)

    # recover_from_ids
    def tran_output(output, vocab):

        output = [o for o in output if o != vocab.get_id("_")]
        output_str = vocab.recover_from_ids(output)
        return "".join(output_str)

    # 解码
    # greedy decoder
    # indices = greedy(y_pred)
    # output_gre = tran_output(indices, vocab)
    # print("greedy:%s"%(output_gre))

    lm_path = "model/aidatatang_200zh.klm"
    # beam search
    beam_size = 300
    indices_beam = beam_search(y_pred, vocab, beam_size, lm_path=lm_path)
    output_beam = tran_output(indices_beam, vocab)
    print("label :%s\tbeam  :%s" %(label, output_beam))
    return output_beam, label

# Character error ratio
import Levenshtein


cers = 0.0
# aidatatang cer=0.32
# test_data = load_dataset("train")
# THCHS-30 cer=0.56
import glob
thchs_path = "test_data/*.npy"
paths = glob.glob(thchs_path)
paths.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))

test_data = []
with open("aidataaudio_test.txt", "r") as t:
    tmp = t.readlines()

for i in range(len(paths)):
    test_data.append([paths[i], list(tmp[i].strip())])

# for path in paths:
#     sample = [path.rstrip(".trn")]
#     with open(path, "r") as f:
#         sample.append(list("".join(f.readline().strip().split())))
#     test_data.append(sample)


import random
datas = random.sample(test_data, 10)
# datas = test_data
n = len(datas)
i = 0
import numpy as np
import time
start_time = time.time()
# f = open("restart_data.txt", "a")
for data in datas:
    # i += 1
    output, label = audio2text(data)
    cer = Levenshtein.distance(output, label)/len(label)
    #if cer > 0.2:
        #print(i)
        #f.write(str(data)+"\n")
    cers += cer/n
    i += 1
    # if i % 1000 == 0:
        # print(cers*n/(i))
    # viz.line(Y=[[np.array(cer)]], X=[i],
    #       opts={"title":"Character error ratio", "legend":["test"]}, win="cer_v4", update="append")

print("Character error ratio : %.2f"%cers)
print("平均用时 : %s"%((time.time()-start_time)/n))
# f.close()
