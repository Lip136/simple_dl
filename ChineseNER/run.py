# encoding:utf-8
'''
1. 加上bert
2. 加上drop_last，也就是可以预测，做法是：在batch_size上padding
'''
import argparse
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






import torch
import torch.optim as optim
from modelsNER import BiLSTM_CRF
from resultCal import calculate
from visdom import Visdom


viz = Visdom(env='NER')
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


class NER():
    def __init__(self, args):
        self.args = args


    def train(self):
        model = BiLSTM_CRF(self.args)
        if self.args.model_path:
            checkpoint = torch.load(self.args.model_path)
            model.load_state_dict(checkpoint.model_state_dict)
            start_epoch = checkpoint.epoch + 1
            best_model = checkpoint.accuracy
        else:
            start_epoch = 1
            best_model = 0

        model.train()
        model = model.to(self.args.device)

        # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


        for epoch in range(start_epoch, self.args.epoch + 1):

            train_loss = []
            for batch_idx, (sentence, tags) in enumerate(self.args.train_loader):

                model.zero_grad()
                # optimizer.zero_grad()

                sentence, tags = sentence.to("cuda"), tags.to("cuda")
                loss = model.neg_log_likelihood(sentence, tags)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.tolist())
                if batch_idx % 30 == 0:
                    print("epoch", epoch, "batch_idx", batch_idx, "loss:", loss.tolist())

            correct = []
            for sentence, tags in self.args.val_loader:
                sentence, tags = sentence.to("cuda"), tags.to("cuda")

                score, predict = model(sentence)
                correct.append(torch.eq(predict, tags).sum().item() / (tags.shape[0] * tags.shape[1]))

            viz.line(Y=[[sum(train_loss) / len(train_loss), 100 * sum(correct) / len(correct)]], X=[epoch],
                     opts=dict(title='loss&acc', legend=['train loss', 'valid acc.']),
                     win='train', update='append')

            if sum(correct) / len(correct) > best_model:
                best_model = sum(correct) / len(correct)

                # path_name = "./model/Bosondata_model" + str(epoch) + ".bin"
                path_name = "./model/Bosondata_model_dict.bin"
                print('准确率：{:.2f}%'.format(100 * best_model))

                self.args.model_state_dict = model.state_dict()
                self.args.accuracy = best_model
                torch.save(self.args, path_name)
                print("model has been saved")

    def test(self):
        # model = torch.load(self.args.model_path, map_location="cuda:0")
        model = BiLSTM_CRF(self.args)
        if self.args.model_path:
            checkpoint = torch.load(self.args.model_path)
            model.load_state_dict(checkpoint.model_state_dict)
            start_epoch = checkpoint.epoch + 1
            best_model = checkpoint.accuracy
        model.eval()
        model = model.to(self.args.device)
        correct = []
        for batch_idx, (sentence, tags) in enumerate(self.args.test_loader):
            sentence, tags = sentence.to("cuda"), tags.to("cuda")
            score, predict = model(sentence)
            # correct.append(torch.eq(predict, tags).sum().item() / (tags.shape[0] * tags.shape[1]))
            entityres = calculate(sentence, predict, id2word, id2tag)
            entityall = calculate(sentence, tags, id2word, id2tag)
            print(entityres, '\n', len(entityall))
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

    def predict(self, ori_sentence):
        model = torch.load(self.args.model_path, map_location="cuda:0")# 有问题
        sentence = []

        # 将sentence变成32*60的tensor
        # word2id
        # for word in ori_sentence:
        #     sentence.append(word2id[word])
        # # 将sentence完全padding
        sentence = torch.cat((torch.tensor(sentence).unsqueeze(dim=0), torch.full((batch_size, 60), 0).long()), dim=0).to("cuda")
        # sentence = torch.tensor(sentence).unsqueeze(dim=0).to("cuda")
        score, predict = model(sentence)
        # 将predict降维1*60
        predict = predict[0].unsqueeze(dim=0)
        entityres = calculate(sentence, predict, id2word, id2tag)
        for entity in entityres[0][0]:
            print("".join([w.split("/")[0] for w in entity]), entity[0].split("/")[1][2:])



def main():
    parser = argparse.ArgumentParser()

    ##

    parser.add_argument("--epoch", default=5, type=int,
                        help="")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_length", default=60, type=int,
                        help="")
    parser.add_argument("--embedding_dim", default=100, type=int,
                        help="")
    parser.add_argument("--hidden_dim", default=200, type=int,
                        help="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available.")

    parser.add_argument("--data_path", default="./data/boson/Bonsondata.tsv", type=str,
                        help="数据")
    parser.add_argument("--model_path", default="./model/Bosondata_model_dict.bin", type=str,
                        help="接着训练")#"./model/Bosondata_model_dict.bin"

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    db_train = GetBatch.NERDataset(args.data_path, "train", args.max_length)
    db_val = GetBatch.NERDataset(args.data_path, "val", args.max_length)
    args.word2id = db_train.word2id
    args.label2id = db_train.label2id
    args.vocab_size = len(args.word2id) + 1

    args.train_loader = data.DataLoader(db_train, batch_size=batch_size, drop_last=True,
                                   shuffle=True)
    args.val_loader = data.DataLoader(db_val, batch_size=batch_size, drop_last=True,
                                 shuffle=True)


    model = NER(args)

if __name__ == "__main__":

    main()

