# encoding:utf-8
'''
1. 加上bert
2. 加上drop_last，也就是可以预测，做法是：在batch_size上padding
'''
import argparse
import GetBatch
import torch.utils.data as data
import pickle
import json


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


import torch
import torch.optim as optim
from modelsNER import BiLSTM_CRF
from visdom import Visdom

#############
# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 200


# tag2id[START_TAG] = len(tag2id)
# tag2id[STOP_TAG] = len(tag2id)
# # print(tag2id)
# id2tag[len(id2tag)] = START_TAG
# id2tag[len(id2tag)] = STOP_TAG
# print(id2tag)

import os
import vocab
class NER():

    def __init__(self, config):
        self.device = config["device"]

        self.datafile = config["train_data"]
        self.vocab_path = config["vocab"]
        self.voc = vocab.Vocab()
        if not os.path.exists(self.vocab_path):
            self.prepare()
        else:
            with open(self.vocab_path, "rb") as f:
                self.voc = pickle.load(f)

        self.model = BiLSTM_CRF(config, self.voc).to(self.device)


    def prepare(self):

        self.voc = vocab.Vocab(self.datafile)
        with open(self.vocab_path, "wb") as f:
            pickle.dump(self.voc, f)

    def train(self):

        viz = Visdom(env='NER')

        self.model.train()

        # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        for epoch in range(self.start_epoch, self.args.epoch + 1):
            train_loss = []
            for batch_idx, (sentence, tags) in enumerate(self.args.train_loader):

                self.model.zero_grad()
                # optimizer.zero_grad()

                sentence, tags = sentence.to(self.args.device), tags.to(self.args.device)
                loss = self.model.neg_log_likelihood(sentence, tags)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.tolist())
                if batch_idx % 30 == 0:
                    print("epoch", epoch, "batch_idx", batch_idx, "loss:", loss.tolist())

            correct = []
            for sentence, tags in self.args.val_loader:
                sentence, tags = sentence.to(self.args.device), tags.to(self.args.device)

                score, predict = self.model(sentence)
                correct.append(torch.eq(predict, tags).sum().item() / (tags.shape[0] * tags.shape[1]))

            viz.line(Y=[[sum(train_loss) / len(train_loss), 100 * sum(correct) / len(correct)]], X=[epoch],
                     opts=dict(title='loss&acc', legend=['train loss', 'valid acc.']),
                     win='train', update='append')

            if sum(correct) / len(correct) > self.best_model:
                self.best_model = sum(correct) / len(correct)

                # path_name = "./model/Bosondata_model" + str(epoch) + ".bin"
                path_name = "./model/Bosondata_model_dict.bin"
                print('准确率：{:.2f}%'.format(100 * self.best_model))

                self.args.model_state_dict = self.model.state_dict()
                self.args.accuracy = self.best_model
                torch.save(self.args, path_name)
                print("model has been saved")


def main():

    config = json.load(open("NER.json", "r"))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    config["device"] = device


    db_train = GetBatch.NERDataset(args.data_path, "train", args.max_length)
    db_val = GetBatch.NERDataset(args.data_path, "val", args.max_length)

    args.train_loader = data.DataLoader(db_train, batch_size=batch_size, drop_last=True,
                                   shuffle=True)
    args.val_loader = data.DataLoader(db_val, batch_size=batch_size, drop_last=True,
                                 shuffle=True)

    model = NER(config)

if __name__ == "__main__":

    main()

