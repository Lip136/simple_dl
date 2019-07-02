# encoding:utf-8

import torch
import torch.optim as optim
import torch.nn as nn

from layers import EncoderRNN, DecoderRNN
from utils import word_dict
import train_model



# Configure training/optimization
model_name = 'cb_model'
attn_model = 'dot' #'general', 'concat'
hidden_size = 500
encoder_n_layers = 2
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

PAD_token = 0 # padding
SOS_token = 1 # start
EOS_token = 2 # end

voc = word_dict.Voc()

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
# Initialize encoder & decoder models
encoder = EncoderRNN.EncoderGRU(hidden_size, embedding, encoder_n_layers, dropout)
decoder = DecoderRNN.DecoderGRU(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


# Run training iterations
print("Starting Training!")
train_model.trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
