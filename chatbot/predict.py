# encoding:utf-8

import torch
import os
import json
from utils import vocab, dataset
import pickle

# Configure training/optimization

attn_model = 'dot' #'general', 'concat'
encoder_n_layers = 1
decoder_n_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Chatbot(object):
    """
    训练模型包含了三个数据集合：训练数据、验证数据、测试数据
    我们把训练和预测，也就是实际进入工程的方式分开
    """

    def __init__(self, config):

        # 文件位置
        self.save_dir = config["model_path"]
        self.model_name = config["net"]["module_name"]
        self.corpus_name = config["task_mode"]

        # 加载模型

        directory = os.path.join(self.save_dir, self.model_name, self.corpus_name, "{}_{}_{}".format(
            encoder_n_layers, decoder_n_layers, config["net"]["hidden_dim"]
        ))
        self.save_file = config["model_file"]
        self.checkpoint = torch.load(os.path.join(directory, self.save_file))

        self.encoder = self.checkpoint['en']
        self.decoder = self.checkpoint['de']

        # if next(self.encoder.parameters()).is_cuda:
        #     print("在GPU上")
        print('Models built and ready to go!')
        # 网络参数


        self.MAX_LENGTH = config["MAX_LENGTH"]
        # 加载数据
        self.handleData = dataset.ChatDataset()
        self.vocab_path = config["vocab"]
        self.voc = vocab.Vocab()
        with open(self.vocab_path, "rb") as f:
            self.voc = pickle.load(f)
        self.SOS_token = self.voc.get_id("<start>")


    # TODO: eval
    def GreedySearchDecoder(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * self.SOS_token

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # 解码:现在采用的是贪婪算法,是否可以改为beam search?
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

    def evaluate(self, sentence):
        max_length = self.MAX_LENGTH
        # words -> indexes
        indexes = self.handleData.indexesFromSentence(self.voc, sentence)
        # Create lengths tensor
        lengths = torch.tensor([len(indexes)])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.tensor(indexes).unsqueeze(1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        # TODO:加入searcher
        # tokens, scores = searcher(input_batch, lengths, max_length)
        tokens, scores = self.GreedySearchDecoder(input_batch, lengths, max_length)

        # indexes -> words
        decoded_words = [self.voc.get_token(token.item()) for token in tokens]
        return decoded_words

    def evaluateInput(self):
        while (1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                # 将句子变成一个id列表
                import jieba
                input_sentence = jieba.cut(input_sentence)
                output_words = self.evaluate(input_sentence)
                # Format and print response sentence
                output_words = [x for x in output_words if x not in ['<pad>', '<start>', "<end>", '<unk>']]
                print('Bot:', ''.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

    def predict_seq(self, input_sentence):

        # Normalize sentence
        # 将句子变成一个id列表
        import jieba
        input_sentence = jieba.cut(input_sentence)
        output_words = self.evaluate(input_sentence)
        # Format and print response sentence
        output_words = [x for x in output_words if x not in ['<pad>', '<start>', "<end>", '<unk>']]
        # print('Bot:', ''.join(output_words))
        return ''.join(output_words)

    def evalModel(self, input_seq):

        # Set dropout layers to eval mode
        self.encoder.eval()
        self.decoder.eval()

        # Begin chatting (uncomment and run the following line to begin)
        # self.evaluateInput()
        result = self.predict_seq(input_seq)
        return result

def main():

    config = json.load(open("chatbot.json", "r"))


    model = Chatbot(config)
    # Run chatting
    print("Starting chatting!")
    # input_seq = "你好"
    # result = model.evalModel(input_seq)
    # print(result)
    model.evaluateInput()

if __name__ == "__main__":
    main()


