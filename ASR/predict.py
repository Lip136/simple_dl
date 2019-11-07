# encoding=utf-8

import pickle, torch
import librosa, os
import ctcdecode
from dataset import ASRDataset
from vocab import Vocab
import random
# load model and vocab
vocab = pickle.load(open("THCHS-30.vocab", "rb"))
model = torch.load("model/asr_gru.bin",  map_location="cpu")
model.eval()

def handleData(audio_path):
    data_manager = ASRDataset()
    a_mfcc = data_manager.load_and_trim(audio_path)
    return a_mfcc

from utils import standardDate

import glob
data_path = glob.glob(os.path.join("data", "*.wav"))
audio_path = random.choice(data_path)
with open(audio_path + ".trn", 'r') as f:
    audio_label = f.readline()

print(audio_label)
a_mfcc = handleData(audio_path)
X = torch.from_numpy(a_mfcc).unsqueeze(dim=0)
X = standardDate(X, vocab)
# 预测
y_pred = model(X)


# 解码
# y_pred = y_pred.transpose(1, 2)
y_pred = y_pred.transpose(1, 0)
def tran_output(output_str):
    output = []
    for i in range(1, len(output_str)):
        if output_str[i] == output_str[i-1] or output_str[i] == "_":
            continue
        output.append(output_str[i])
    return output

vocab_list = list(vocab.token2id.keys())

assert y_pred.size(2) == len(vocab_list)
beam_size = 10

lm_path = "THCHS-30.arpa"
decoder = ctcdecode.CTCBeamDecoder(vocab_list, beam_width=beam_size,
                                           blank_id=vocab.get_id("_"),model_path=lm_path)


# print(y_pred.shape)
y_pred = y_pred.softmax(dim=2)
beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(y_pred)
# print(beam_result[0][0][0:3], out_seq_len)
# print(beam_scores)
# output_str = convert_to_string(beam_result[0][0], vocab[1], out_seq_len[0][0])

output_str = vocab.recover_from_ids(beam_result[0][0][0:out_seq_len[0][0]].tolist())
output_beam = "".join(output_str)



# greedy decoder
y_pred = y_pred.squeeze()
indices = y_pred.topk(1)[1][:, 0]
output_str = vocab.recover_from_ids(indices.tolist())
output_gre = "".join(tran_output(output_str))
print("beam  :%s \ngreedy:%s"%(output_beam,output_gre))
print(output_beam == output_gre)



# my beam search decode
def my_beam_decode(prob, beam_width):
    # prob.shape = (seq_len, n_class)
    # 先通过score得到index
    score, start_index = torch.topk(prob[0], beam_width)
    result = [start_index]
    for i in range(1, len(prob)):
        tmp_p = torch.cat([prob[i] + s for s in score])
        score, index = torch.topk(tmp_p, beam_width)
        result.append(index)
    # print((torch.cat(result) >= 0).sum())
    # 解析index
    next_index = result[-1]
    res = [next_index]
    for i in range(len(result)-2, -1, -1):
        next_index = torch.index_select(result[i], 0, torch.div(next_index, prob.size(1)))
        res.append(next_index)

    result_index = torch.stack([i%10 for i in res[::-1]], dim=0)
    return result_index


indices = my_beam_decode(y_pred, beam_width=beam_size)[:, 0]
output_my_beam = vocab.recover_from_ids(indices.tolist())
output_my_beam = "".join(tran_output(output_my_beam))
print(output_my_beam)