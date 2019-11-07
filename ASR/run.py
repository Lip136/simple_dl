# encoding=utf-8

from layer import ASR_cnn, ASR_gru
import pickle, os

from torch import nn, optim
import torch

batch_size = 64
hidden_dim = 128
mfcc_dim = 13

from dataset import ASRDataset
from vocab import Vocab
# from warpctc_pytorch import CTCLoss

from visdom import Visdom
viz = Visdom(env="asr")
# python -m visdom.server

config = {
    "vocab_path" : "./",
    "task" : "THCHS-30",
    "data_path" : "data",
    "model_path" : "./"
}
from utils import standardDate

device = "cuda" if torch.cuda.is_available() else "cpu"
class ASR(object):

    def __init__(self, config):
        self.vocab_path = config["vocab_path"]
        self.task = config["task"]
        self.data_path = config["data_path"]
        self.vocab = self.prepare()
        self.data_manager = ASRDataset(self.data_path)

        # self.model = ASR_cnn(mfcc_dim, hidden_dim, self.vocab.size()).to(device)
        self.model = ASR_gru(mfcc_dim, hidden_dim, self.vocab.size()).to(device)
        self.criterion = nn.CTCLoss()
        # self.criterion = CTCLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.min_loss = 1000.0
        self.model_path = config["model_path"]

    def prepare(self):
        # 加载词表
        vocab_file = os.path.join(self.vocab_path, self.task + ".vocab")
        if os.path.exists(vocab_file):
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
        else:
            import glob
            filenames = glob.glob(os.path.join(self.data_path, "*.trn"))
            vocab = Vocab(filenames, initial_tokens=["_"])
            with open(vocab_file, "wb") as f:
                pickle.dump(vocab, f)

        print("一共有%s个词语"%vocab.size())

        return vocab

    def train(self):

        for epoch in range(50):
            print_loss = []
            batch_data = self.data_manager.gen_mini_batches(mode="train", batch_size=batch_size, voc=self.vocab)
            for batch in batch_data:
                self.optimizer.zero_grad()

                X, X_length, Y, Y_length = batch
                X_length = X_length.to(device)
                Y_length = Y_length.to(device)
                X = standardDate(X, vocab=self.vocab).to(device)

                Y = Y.to(device)
                # padding
                # bs * 13 * seq_len
                # [bs, label_size, seq_len]
                y_pred = self.model(X)

                # y_pred = y_pred.transpose(1, 0)
                # y_pred = y_pred.transpose(2, 0)
                # T, N, C => seq_len, bs, label_class
                # (seq_len*bs*label_size, bs*label_len, )
                # print(y_pred.size(), Y.size(), X_length.size(), Y_length.size())
                # Y = Y.view(-1).int()
                # X_length = X_length.int()
                # X_length = torch.full(size=(y_pred.size(1), ), fill_value=y_pred.size(0)).cuda().int()
                # Y_length = Y_length.int()
                assert y_pred.type() == 'torch.cuda.FloatTensor'
                loss = self.criterion(y_pred, Y, X_length, Y_length)
                print_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()
            train_loss = sum(print_loss)/len(print_loss)
            # print(train_loss)
            val_data = self.data_manager.gen_mini_batches(mode="val", batch_size=batch_size, voc=self.vocab)
            val_loss = []
            with torch.no_grad():
                for batch in val_data:
                    X, X_length, Y, Y_length = batch
                    X_length = X_length.to(device)
                    Y_length = Y_length.to(device)
                    X = standardDate(X, vocab=self.vocab).to(device)

                    Y = Y.to(device)

                    y_pred = self.model(X)
                    # y_pred = y_pred.transpose(1, 0)
                    # y_pred = y_pred.transpose(2, 0)
                    # Y = Y.view(-1).int()
                    # X_length = X_length.int()
                    # Y_length = Y_length.int()
                    loss = self.criterion(y_pred, Y, X_length, Y_length)
                    val_loss.append(loss.item())


            val_l = sum(val_loss)/len(val_loss)
            print("train {}, val {}".format(train_loss, val_l))
            viz.line(Y=[[train_loss, val_l]], X=[epoch],
                    opts=dict(title="loss", legend=["train loss", "valid loss"]),
                    win="train", update="append")

            if val_l < self.min_loss:
                self.min_loss = val_l
                torch.save(self.model.state_dict(), os.path.join(self.model_path, "asr_gru.tar"))
                torch.save(self.model, os.path.join(self.model_path, "asr_gru.bin"))



if __name__ == "__main__":
    asr_model = ASR(config)
    asr_model.train()









