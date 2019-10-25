# encoding:utf-8
'''
训练阶段：1.初始化解码器和编码器
        2.获得batch数据： train()函数是训练的主函数
        3.训练：trainIters()函数   
'''
import torchsnooper
import torch
import torch.optim as optim
import torch.nn as nn
import os
import random

from layers import EncoderRNN, DecoderRNN
import json
import sys
sys.path.append("./utils/")
from utils import vocab, dataset
import pickle
import logging

# Configure training/optimization

# attn_model = 'dot' #'general', 'concat'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Chatbot(object):
    """
    训练模型包含了三个数据集合：训练数据、验证数据、测试数据
    我们把训练和预测，也就是实际进入工程的方式分开
    """

    def __init__(self, config, restart=False):

        # 文件位置
        root_Path = config["data_file_path"]
        self.corpus_name = config["task_mode"]
        self.datafile = os.path.join(root_Path, "qingyun.tsv")

        self.vocab_path = config["vocab"]

        self.voc = vocab.Vocab()
        if not os.path.exists(self.vocab_path):
            self.prepare()
        else:
            with open(self.vocab_path, "rb") as f:
                self.voc = pickle.load(f)

        # 网络参数
        self.net_config = config["net"]

        self.save_dir = config["model_path"]
        self.save_file = config["model_file"]

        self.model_name = config["net"]["module_name"]
        self.restart = restart
        if self.restart:
            directory = os.path.join(self.save_dir, self.model_name, self.corpus_name, "{}_{}_{}".format(
                self.net_config["encoder_n_layers"], self.net_config["decoder_n_layers"], self.net_config["hidden_dim"]
            ))
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            self.checkpoint = torch.load(os.path.join(directory, self.save_file))



        self.learning_rate = config["optimizer"]["learning_rate"]
        self.batch_size = config["batch_size"]

        self.print_every = config["print_num"]
        self.epochs = config["epoch_num"]
        self.MAX_LENGTH = config["MAX_LENGTH"]
        # 加载数据
        self.handleData = dataset.ChatDataset(self.datafile, self.MAX_LENGTH)
        self.SOS_token = self.voc.get_id("<start>")

        # 初始化解码器和编码器
        self.initializeED()


    def prepare(self):

        logger = logging.getLogger(self.corpus_name)
        logger.info('Building vocabulary...')

        self.voc = vocab.Vocab(self.datafile)
        with open(self.vocab_path, "wb") as v:
            pickle.dump(self.voc, v)

    # TODO: training
    # 初始化解码器和编码器
    def initializeED(self):
        logger = logging.getLogger(self.corpus_name)
        logger.info(json.dumps(self.net_config, ensure_ascii=False))
        # Initialize word embeddings
        # 采用的torch自带的Embedding的方式
        # self.embedding = nn.Embedding(self.voc.size(), self.net_config["emb_dim"], padding_idx=self.voc.get_id("<pad>"))
        # 采用加载词向量的方式
        # 出现RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
        # 原因是: numpy默认的float是float64位的,而我们需要的是float32位
        embedding_path = "/home/ptface02/PycharmProjects/data/tencent/Tencent_AILab_ChineseEmbedding.txt"
        self.voc.load_pretrained_embeddings(embedding_path)
        # pre_embedding = torch.from_numpy(self.voc.embeddings).float()
        pre_embedding = self.voc.embeddings
        print(self.voc.size(), pre_embedding.shape)
        self.embedding = nn.Embedding.from_pretrained(pre_embedding, freeze=False)
        # nn.init.kaiming_normal_()凯明大神的初始化方法
        # nn.init.xavier_uniform_()
        # 由于我们有6000多个词语都是正太分布的初始化,所以我们应该换一种方式
        logger.info('Building encoder and decoder ...')
        # print('Building encoder and decoder ...')
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN.EncoderGRU(self.net_config, self.embedding)
        self.decoder = DecoderRNN.DecoderGRU(self.net_config, self.embedding, self.voc.size())
        # 终于明白了,如果你不先设置cuda,那么优化器就会出问题...
        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        # Initialize optimizers
        logger.info("Building optimizers ...")
        # print('Building optimizers ...')
        # 需要加上weight_decay, 默认就是L2
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.learning_rate * self.net_config["decoder_learning_ratio"])
        # self.optimizer = optim.Adam([{"params" : self.encoder.parameters()},
        #                              {"params" : self.decoder.parameters(), "lr":1e-5}],
        #                              , lr=1e-2)
        # TODO:学习率的动态衰减
        # self.encoder_scheduler = optim.lr_scheduler.ExponentialLR(self.encoder_optimizer, gamma=0.9)
        # self.decoder_scheduler = optim.lr_scheduler.ExponentialLR(self.decoder_optimizer, gamma=0.9)
        # Load model if a loadFilename is provided
        if self.restart:
            encoder_sd = self.checkpoint['en']
            decoder_sd = self.checkpoint['de']
            encoder_optimizer_sd = self.checkpoint['en_opt']
            decoder_optimizer_sd = self.checkpoint['de_opt']
            embedding_sd = self.checkpoint['embedding']

            self.embedding.load_state_dict(embedding_sd)
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
            self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        logger.info("")
        # print('Models built and ready to go!')

    #TODO: train model
    # mask loss
    def maskNLLLoss(self, inp, target, mask):
        # (64 * 100023) (64) (64)
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.unsqueeze(1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    # 单次训练迭代
    # 1.teacher_forcing_ratio; 2.gradient clipping
    # @torchsnooper.snoop()
    def train(self, batch_data, clip):

        input_variable, lengths, target_variable, mask, max_target_len = batch_data
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.optimizer.zero_grad()
        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)
        # print("encoder:{} en_hidden:{}".format(encoder_outputs.shape, encoder_hidden.shape))
        # encoder:torch.Size([10, 64, 128]) en_hidden:torch.Size([2, 64, 128])
        # start with SOS tokens for each sentence
        # decoder_input = torch.tensor([[self.SOS_token for _ in range(self.batch_size)]])

        batch_size = encoder_hidden.size(1)
        decoder_input = torch.rand(1, batch_size).fill_(self.SOS_token).long()
        decoder_input = decoder_input.to(device)
        # print("decoder_inp:{}".format(decoder_input.shape))
        # decoder_inp:torch.Size([1, 64])
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # print("decoder_hid:{}".format(decoder_hidden.shape))
        # decoder_hid:torch.Size([2, 64, 128])
        # 其实decoder_hidden == encoder_hidden
        # teacher forcing
        use_teacher_forcing = True if random.random() < self.net_config["teacher_forcing_ratio"] else False

        if use_teacher_forcing:
            for t in range(max_target_len):

                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, encoder_hidden, encoder_outputs
                )
                # (64 * 100023) (2 * 64 * 128)
                # TODO teacher forcing: next input is current target
                decoder_input = target_variable[t].unsqueeze(0) #重点 [1, 64]

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = topi.transpose(0, 1)
                decoder_input = decoder_input.to(device)

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total

        loss.backward()
        # TODO:clip gradients
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # print(self.encoder_optimizer.state_dict())

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # self.optimizer.step()

        return sum(print_losses) / n_totals

    def eval(self, batch_data):
        input_variable, lengths, target_variable, mask, max_target_len = batch_data

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        batch_size = encoder_hidden.size(1)
        decoder_input = torch.rand(1, batch_size).fill_(self.SOS_token).long()
        decoder_input = decoder_input.to(device)
        # print("decoder_inp:{}".format(decoder_input.shape))
        # decoder_inp:torch.Size([1, 64])
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # print("decoder_hid:{}".format(decoder_hidden.shape))
        # decoder_hid:torch.Size([2, 64, 128])
        # 其实decoder_hidden == encoder_hidden
        # teacher forcing
        use_teacher_forcing = True if random.random() < self.net_config["teacher_forcing_ratio"] else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, encoder_hidden, encoder_outputs
                )
                # (64 * 100023) (2 * 64 * 128)
                # TODO teacher forcing: next input is current target
                decoder_input = target_variable[t].unsqueeze(0)  # 重点 [1, 64]

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = torch.tensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(device)

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total


        return sum(print_losses) / n_totals

    def trainIters(self):
        # Load batches for each iteration
        # 选择数据相当于：并没有选择所有的数据，每次随机选batch_size个数据，一共选n_iteration次
        # 如果数据量本来就过大，那肯定一次数据都选不完：10w数据, 选了4000*32=12.8w, 这才训练一个epoch
        # 怪不得我的结果不太好
        logger = logging.getLogger(self.corpus_name)
        # Initializations
        logger.info("Initializaing ...")
        # print("Initializaing ...")
        best_loss = float("inf")
        start_iteration = 1
        print_loss = 0
        # TODO:接着训练
        if self.restart:
            start_iteration = self.checkpoint['epoch'] + 1
            best_loss = self.checkpoint["loss"]
        # Training loop
        logger.info("Training ...")
        # print("Training ...")
        for epoch in range(start_iteration, self.epochs + 1):

            # self.encoder_scheduler.step(epoch)
            # self.decoder_scheduler.step(epoch)

            data_batches = self.handleData.gen_mini_batches("train", self.batch_size, self.voc)

            # Ensure dropout layers are in train mode
            self.encoder.train()
            self.decoder.train()

            num_batch = 0
            for training_batch in data_batches:

                # input_variable, lengths, target_variable, mask, max_target_len = training_batch
                loss = self.train(training_batch, self.net_config["clip"])
                print_loss += loss
                num_batch += 1

            # eval
            dev_batches = self.handleData.gen_mini_batches("dev", self.batch_size, self.voc)
            # Set dropout layers to eval mode
            self.encoder.eval()
            self.decoder.eval()
            dev_num_batch = 0
            dev_print_loss = 0
            with torch.no_grad():
                for dev_batch in dev_batches:

                    dev_loss = self.eval(dev_batch)
                    dev_print_loss += dev_loss
                    dev_num_batch += 1

            dev_loss_avg = dev_print_loss / (self.print_every * dev_num_batch)

            # 每print_every打印一次loss数据
            if epoch % self.print_every == 0:
                print_loss_avg = print_loss / (self.print_every * num_batch)

                logger.info("epoch:{}\tPercent complete:{:.1f}%\tAverage loss:{:.4f}\tAverage dev loss:{:.4f}".format(
                     epoch, epoch / self.epochs * 100, print_loss_avg, dev_loss_avg
                ))
                # print("epoch:{}\tPercent complete:{:.1f}%\tAverage loss:{:.4f}\tAverage dev loss:{:.4f}".format(
                #     epoch, epoch / self.epochs * 100, print_loss_avg, dev_loss_avg
                # ))
                print_loss = 0


            # 每save_every保存一次模型
            if dev_loss_avg < best_loss:

                best_loss = dev_loss_avg

                directory = os.path.join(self.save_dir, self.model_name, self.corpus_name, "{}_{}_{}".format(
                    self.net_config["encoder_n_layers"], self.net_config["decoder_n_layers"], self.net_config["hidden_dim"]
                ))

                if not os.path.exists(directory):
                    os.makedirs(directory)
                # print(self.encoder_optimizer.state_dict())
                torch.save({
                    'epoch': epoch,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': best_loss,
                    "embedding": self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

                torch.save({'en': self.encoder,
                            'de': self.decoder
                            }, os.path.join(directory, 'infer{}_{}.bin'.format(epoch, 'checkpoint')))


def main():

    config = json.load(open("chatbot.json", "r"))

    log_path = config["log_path"]
    corpus_name = config["task_mode"]


    logger = logging.getLogger(corpus_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model = Chatbot(config, restart=False)
    # Run training iterations
    logger.info("Starting Training!")
    # print("Starting Training!")
    model.trainIters()

if __name__ == "__main__":
    main()


