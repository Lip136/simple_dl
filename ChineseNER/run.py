# encoding:utf-8
import GetBatch
import torch.utils.data as data
import pickle
import numpy as np

with open('./data/boson/Bosondata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
print("train len:{}\ntest len:{}\nvalid len:{}".format(len(x_train), len(x_test), len(x_valid)))
batch_size = 32
train_loader = data.DataLoader(GetBatch.MyDataset(x_train, y_train), batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = data.DataLoader(GetBatch.MyDataset(x_valid, y_valid), batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = data.DataLoader(GetBatch.MyDataset(x_test, y_test), batch_size=batch_size, drop_last=True)


import torch
import torch.optim as optim
from modelsNER import BiLSTM_CRF
from resultCal import calculate
from visdom import Visdom

viz = Visdom(env='NER')
#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 50


tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)
# print(tag2id)
id2tag[len(id2tag)] = START_TAG
id2tag[len(id2tag)] = STOP_TAG
print(id2tag)
# model = BiLSTM_CRF(batch_size, len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)
# model = model.to("cuda")
# best_model = 0
# # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# for epoch in range(1, EPOCHS+1):
#
#     train_loss = []
#     for batch_idx, (sentence, tags) in enumerate(train_loader):
#
#         model.zero_grad()
#         # optimizer.zero_grad()
#
#         # sentence = torch.tensor(sentence, dtype=torch.long)
#         # tags = torch.tensor([tag2id[t] for t in tags.numpy()], dtype=torch.long)
#         # print(tags)
#         sentence, tags = sentence.to("cuda"), tags.to("cuda")
#         loss = model.neg_log_likelihood(sentence, tags)
#
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.tolist())
#         if batch_idx % 30 == 0:
#             print("epoch", epoch, "batch_idx", batch_idx, "loss:", loss.tolist())
#
#
#
#     correct = []
#     for sentence, tags in val_loader:
#         sentence, tags = sentence.to("cuda"), tags.to("cuda")
#
#         score, predict = model(sentence)
#         correct.append(torch.eq(predict, tags).sum().item()/(tags.shape[0]*tags.shape[1]))
#
#     viz.line(Y=[[sum(train_loss)/len(train_loss), 100 * sum(correct)/len(correct)]], X=[epoch],
#              opts=dict(title='loss&acc', legend=['train loss', 'valid acc.']),
#              win='train', update='append')
#
#
#     if sum(correct)/len(correct) > best_model:
#         best_model = sum(correct)/len(correct)
#
#         path_name = "./model/Bosondata_model" + str(epoch) + ".pkl"
#         print('准确率：{:.2f}%'.format(100 * best_model))
#         torch.save(model, path_name)
#         print("model has been saved")

def evaluable(model_path):
    model = torch.load(model_path, map_location="cuda:0")
    correct = []
    for batch_idx, (sentence, tags) in enumerate(test_loader):


        sentence, tags = sentence.to("cuda"), tags.to("cuda")
        score, predict = model(sentence)
        # correct.append(torch.eq(predict, tags).sum().item() / (tags.shape[0] * tags.shape[1]))
        entityres = calculate(sentence, predict, id2word, id2tag)
        entityall = calculate(sentence, tags, id2word, id2tag)
        print(entityres, '\n', entityall)
        break
        # for i in range(batch_size):
        #     entityres = calculate(sentence[i], predict[i], id2word, id2tag, entityres)
        #     entityall = calculate(sentence[i], tags[i], id2word, id2tag, entityall)
        # jiaoji = [i for i in entityres if i in entityall]
        # if len(jiaoji) != 0:
        #     zhun = float(len(jiaoji)) / len(entityres)
        #     zhao = float(len(jiaoji)) / len(entityall)
        #     print("test:")
        #     print("准确率:", zhun)
        #     print("召回率:", zhao)
        #     print("f:", (2 * zhun * zhao) / (zhun + zhao))
        # else:
        #     print("准确率:", 0)
#
model_path = "./model/Bosondata_model10.pkl"
evaluable(model_path)
