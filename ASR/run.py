# encoding=utf-8

import pickle, os, json
from torch import nn, optim
import torch
from lookahead import Lookahead
# from layer import ASR_cnn, ASR_gru
# from layer_new import ASR_gru
from layer_v9 import ASR_lstm
from dataset import ASRDataset
from utils import standardDate, load_dataset
from vocab import Vocab
# from warpctc_pytorch import CTCLoss
import numpy as np
from visdom import Visdom
import time
viz = Visdom(env="asr", server='http://10.3.27.36', port=8097)
assert viz.check_connection()
# python -m visdom.server
from tensorboardX import SummaryWriter


torch.backends.cudnn.benchmark = True
# 清华的句子平均53个字
# 数据堂句子平均10个字
class ASR(object):

    def __init__(self, config):
        self.task = config["task_mode"]
        self.vocab_path = config["path"]["vocab"]
        self.model_path = config["path"]["model"]
        self.data_path = config["path"]["data"]

        self.mfcc_dim = config["mfcc_dim"]
        self.vocab = self.prepare()
        self.data_manager = ASRDataset("./")
        self.batch_size = config["batch_size"]

        self.epochs = config["epoch_num"]
        self.print_epoch = config["print_num"]

        self.model = ASR_lstm(self.mfcc_dim, config["net"]["hidden_dim"], self.vocab.size(), device, num_layers=2).to(device)
        print(self.model)
        tmp = torch.zeros(8, 39, 410).to(device)
        with SummaryWriter(comment='asr') as w:
            w.add_graph(self.model, tmp)

        # torch.cuda.empty_cache()
        # 多GPU训练
        # if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            # self.model = nn.DataParallel(self.model)

        self.criterion = nn.CTCLoss() #zero_infinity=True
        # self.criterion = CTCLoss()
        opt_para = config["optimizer"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt_para["lr"], weight_decay=opt_para["weight_decay"])
        self.lookahead = Lookahead(self.optimizer, k=5, alpha=0.5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt_para["scheduler_step_size"], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.min_loss = 1000.0


    def prepare(self):
        if not os.path.exists(self.vocab_path):
            os.mkdir(self.vocab_path)
        # 加载词表
        vocab_file = os.path.join(self.vocab_path, self.task + ".vocab")
        if os.path.exists(vocab_file):
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
        else:
            filenames = []
            for dirpath, dirfiles, filename in os.walk(os.path.join(self.data_path,"train")):
                for name in filename:
                    if os.path.splitext(name)[-1] == ".trn":
                        filenames.append(os.path.join(dirpath, name))
            vocab = Vocab(self.mfcc_dim, filenames, ["_"])
            with open(vocab_file, "wb") as f:
                pickle.dump(vocab, f)

        print("一共有%s个词语"%vocab.size())

        return vocab

    def train(self, restart=False):
        if restart:
            model_path = os.path.join(self.model_path, "asr%s_v9.bin"%self.task)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["model"])
            self.min_loss = checkpoint["min_loss"]
        # 30 epoch 就已经无法下降了
        print("起始loss:%.2f"%self.min_loss)
        writer = SummaryWriter("epoch_data")
        for epoch in range(self.epochs):
            print_loss = 0.0
            train_bs_epoch = 0
            batch_data = self.data_manager.gen_mini_batches(mode="train", batch_size=self.batch_size, voc=self.vocab)
            # writer_bs = SummaryWriter("batch_data")
            for batch in batch_data:

                X, X_length, Y, Y_length, max_target_len = batch
                X_length = X_length.to(device)
                Y_length = Y_length.to(device)
                X = standardDate(X, vocab=self.vocab).to(device)
                Y = Y.to(device)
                # print(X.shape)
                # if X.size(0) != self.batch_size:
                    # continue

                # self.optimizer.zero_grad()
                self.lookahead.zero_grad()
                y_pred = self.model(X)
                # 这可以检测一遍数据
                if torch.any(torch.isnan(y_pred)):
                    print("输出数据有问题")
                    np.save("out_data_nan_%s.npy"%epoch, np.array(y_pred.cpu().tolist(), dtype=np.float32))
                    break
                # np.save("error_pred.npy", np.array(y_pred.cpu().tolist()))
                # y_pred = torch.cat(y_pred.chunk(torch.cuda.device_count(), dim=0), dim=1)
                # y_pred = y_pred.transpose(0, 1)
                loss = self.criterion(y_pred, Y, X_length, Y_length)
                train_bs_epoch += 1
                print_loss += loss.item()
                # print(loss.item())

                viz.line(Y=[[loss.item()]], X=[np.array(train_bs_epoch)],
                         opts=dict(title="batch高一点", legend=["train loss"]),
                         win="step_lr", update="append")
                # writer_bs.add_scalar("train loss", loss.item(), train_bs_epoch)

                loss.backward()
                # _ = nn.utils.clip_grad_norm_(self.model.parameters(), 50.0)
                # self.optimizer.step()
                self.lookahead.step()
            # writer_bs.close()
            train_loss = print_loss/train_bs_epoch
            torch.cuda.empty_cache()
            
            val_data = self.data_manager.gen_mini_batches(mode="val", batch_size=self.batch_size, voc=self.vocab)
            val_loss = 0.0
            with torch.no_grad():
                val_bs_epoch = 0
                for batch in val_data:
                    X, X_length, Y, Y_length, max_target_len = batch
                    X_length = X_length.to(device)
                    Y_length = Y_length.to(device)
                    X = standardDate(X, vocab=self.vocab).to(device)
                    Y = Y.to(device)

                    # if X.size(0) != self.batch_size:
                        # continue

                    y_pred = self.model(X)
                    # y_pred = torch.cat(y_pred.chunk(torch.cuda.device_count(), dim=0), dim=1)
                    # y_pred = y_pred.transpose(0, 1)
                    loss = self.criterion(y_pred, Y, X_length, Y_length)
                    val_bs_epoch += 1
                    val_loss += loss.item()

            val_l = val_loss/val_bs_epoch
            self.scheduler.step()

            viz.line(Y=[[train_loss, val_l]], X=[epoch],
                     opts=dict(title="epoch能否提高", legend=["train loss", "valid loss"]),
                     win="step_lr_epoch", update="append")
            # tensorboard
            writer.add_scalars('epoch', {"train loss": train_loss,
                                         "valid loss": val_l,
                                        }, epoch)

            if epoch % self.print_epoch == 0:
                # self.scheduler.get_lr()[0]
                print("epoch {:0>2d}, train {:.2f}, val {:.2f}, lr {}, now time {}".format(
                    epoch, train_loss, val_l, self.optimizer.param_groups[0]['lr'],
                    time.strftime("%m-%d %H:%M", time.localtime(time.time()))))

            torch.cuda.empty_cache()
            if val_l < self.min_loss:
                self.min_loss = val_l
                torch.save({"model":self.model.state_dict(),
                            "min_loss":self.min_loss},
                           os.path.join(self.model_path, "asr%s_v9.bin"%self.task))

            # torch.cuda.empty_cache()
        writer.close()


if __name__ == "__main__":
    config = json.load(open("asr.json", 'r'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    asr_model = ASR(config)
    asr_model.train(restart=False)









