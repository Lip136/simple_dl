# encoding:utf-8
import torchsnooper
import torch
import torch.optim as optim
import torch.nn as nn
import os
import random

from layers import EncoderRNN, DecoderRNN
from utils import preprocess, HandleData

# Configure training/optimization
model_name = 'cb_model'
attn_model = 'dot' #'general', 'concat'
hidden_size = 500
encoder_n_layers = 1
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10



# corpus_name = "cornell movie-dialogs corpus"
# corpus = os.path.join("/home/ptface02/PycharmProjects/documents/simple_dl-git/chatbot/data", corpus_name)
# datafile = os.path.join(corpus, "formatted_movie_lines.txt")


corpus_name = "xiaohuangji-dialogs corpus"
corpus = os.path.join("/media/ptface02/H1/dataSet/中文聊天语料/chaotbot_corpus_Chinese/clean_chat_corpus")
datafile = os.path.join(corpus, "xiaohuangji.tsv")

save_dir = os.path.join("/home/ptface02/PycharmProjects/documents/simple_dl-git/chatbot/data", "model")

# loadFilename = None

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}_{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))
checkpoint = torch.load(loadFilename)

class PrepareDate():
    def __init__(self, SOS_token = 1):
        self.handleData = HandleData.BatchManager()
        # 加载数据
        self.preprocessData = preprocess.DataSet(datafile, corpus_name)
        self.voc, self.pairs = self.preprocessData.loadPrepareData()
        self.SOS_token = SOS_token

    def exampleBatch(self):
        # Example for validation
        small_batch_size = 5
        batches = self.handleData.batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(small_batch_size)])
        input_variable, lengths, target_variable, mask, max_target_len = batches
        print("input_variable:", input_variable)
        print("lengths:", lengths)
        print("target_variable:", target_variable)
        print("mask:", mask)
        print("max_target_len:", max_target_len)

    #TODO: train model
    # mask loss
    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    # 单次训练迭代
    # 1.teacher_forcing_ratio; 2.gradient clipping
    def train(self, input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
              enmedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
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
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
        # start with SOS tokens for each sentence
        decoder_input = torch.LongTensor([[self.SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # teacher forcing
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, encoder_hidden, encoder_outputs
                )
                # TODO teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)

                mask_loss, n_Total = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_Total)
                n_totals += n_Total

        loss.backward()
        # TODO:clip gradients
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def trainIters(self, model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename):
        # Load batches for each iteration
        training_batches = [self.handleData.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                            for _ in range(n_iteration)]
        # Initializations
        print("Initializaing ...")
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training ...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
            print_loss += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration:{}\tPercent complete:{:.1f}%\tAverage loss:{:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg
                ))
                print_loss = 0

            if iteration % save_every == 0:
                directory = os.path.join(save_dir, model_name, corpus_name, "{}_{}_{}".format(
                    encoder_n_layers, decoder_n_layers, hidden_size
                ))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    "voc_dict": voc.__dict__,
                    "embedding": embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

    # TODO: training
    # 初始化解码器和编码器
    def initializeED(self):
        # Set checkpoint to load from; set to None if starting from scratch
        # checkpoint_iter = 4000
        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}_{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cuda'))

        # Load model if a loadFilename is provided
        if loadFilename:
            # If loading on same machine the model was trained on
            # checkpoint = torch.load(loadFilename)
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            self.voc.__dict__ = checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # Initialize word embeddings
        self.embedding = nn.Embedding(self.voc.num_words, hidden_size)
        if loadFilename:
            self.embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN.EncoderGRU(hidden_size, self.embedding, encoder_n_layers, dropout)
        self.decoder = DecoderRNN.DecoderGRU(attn_model, self.embedding, hidden_size, self.voc.num_words, decoder_n_layers, dropout)
        if loadFilename:
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        print('Models built and ready to go!')

    # training
    # @torchsnooper.snoop()
    def StartTrain(self):
        # 初始化
        self.initializeED()
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()
        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

        # Run training iterations
        print("Starting Training!")
        self.trainIters(model_name, self.voc, self.pairs, self.encoder, self.decoder, encoder_optimizer, decoder_optimizer,
                        self.embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                               print_every, save_every, clip, corpus_name, loadFilename)

    # TODO: eval

    def GreedySearchDecoder(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * self.SOS_token

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        # words -> indexes
        indexes_batch = [self.handleData.indexesFromSentence(self.voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        # TODO:加入searcher
        # tokens, scores = searcher(input_batch, lengths, max_length)
        tokens, scores = self.GreedySearchDecoder(input_batch, lengths, max_length)

        # indexes -> words
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def evaluateInput(self):
        input_sentence = ''
        while (1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = self.preprocessData.seg_sentence(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ''.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")


    def evalModel(self):
        # 初始化参数
        self.initializeED()
        # Set dropout layers to eval mode
        self.encoder.eval()
        self.decoder.eval()
        # Initialize search module
        # searcher = self.GreedySearchDecoder(input_seq, input_length, max_length)

        # Begin chatting (uncomment and run the following line to begin)
        self.evaluateInput()


if __name__ == "__main__":
    model = PrepareDate()
    # model.StartTrain()
    model.evalModel()

