# encoding:utf-8

import torch
import torch.nn as nn

import random
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = "SOS"
teacher_forcing_ratio = 1.0


# mask loss
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# 单次训练迭代
# 1.teacher_forcing_ratio; 2.gradient clipping
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
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
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
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

            mask_loss, n_Total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
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

            mask_loss, n_Total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
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

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]
    # Initializations
    print("Initializaing ...")
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training ...")
    for iteration in range(start_iteration, n_iteration+1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration:{}\tPercent complete:{:.1f}%\tAverage loss:{:.4f}".format(
                iteration, iteration/n_iteration * 100, print_loss_avg
            ))
            print_loss = 0

        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name, "{}_{}_{}".format(
                encoder_n_layers, decoder_n_layers, hidden_size
            ))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'iteration':iteration,
                'en':encoder.state_dict(),
                'de':decoder.state_dict(),
                'en_opt':encoder_optimizer.state_dict(),
                'de_opt':decoder_optimizer.state_dict(),
                'loss':loss,
                "voc_dict":voc.__dict__,
                "embedding":embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

