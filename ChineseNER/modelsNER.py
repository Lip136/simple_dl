# encoding:utf-8
import torch
import torch.nn as nn
import torchsnooper

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"



class BiLSTM_CRF(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        self.tag_to_ix = args.label2id
        self.tagset_size = len(args.label2id)
        self.batch_size = args.batch_size

        self.word_embeds = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0) # padding_idx=0
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(args.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        t = torch.randn(args.batch_size, self.tagset_size, self.tagset_size)
        self.transitions = nn.Parameter(
            nn.init.kaiming_normal_(t))


        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[:, self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, :, self.tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        w1 = torch.randn(2, self.batch_size, self.hidden_dim // 2).to("cuda")
        w2 = torch.randn(2, self.batch_size, self.hidden_dim // 2).to("cuda")
        return (nn.init.kaiming_normal_(w1),
                nn.init.kaiming_normal_(w2))

    # @torchsnooper.snoop()
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentence = sentence.transpose(0, 1)
        embeds = self.word_embeds(sentence) # len(sentence) * batch_size * embedding_dim
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        # lstm_feats = self.hidden2tag(torch.cat([self.hidden[-2], self.hidden[-1]], dim=1))
        return lstm_feats
    # @torchsnooper.snoop()
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.batch_size, self.tagset_size), -10000.).to("cuda")

        init_vvars[:, self.tag_to_ix[START_TAG]] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # 每一个feat的shape是(32, 22), 每一个feats.shape=(60, 32, 22)
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[:, next_tag, :] # 32*22
                # 最大的tag的index
                # best_tag_id = torch.argmax(next_tag_var, dim=1) # 128
                # bptrs_t.append(best_tag_id) # 22 × 128 每一个

                # viter = torch.tensor([next_tag_var[i][best_tag_id[i]] for i in range(self.batch_size)], dtype=torch.float)
                # viter = torch.gather(next_tag_var, dim=1, index=best_tag_id.unsqueeze(dim=1)).squeeze(dim=1)
                # viterbivars_t.append(viter) # 22 × 128

                best_tag, best_tag_id= next_tag_var.max(dim=1)
                bptrs_t.append(best_tag_id) # 22 * 32
                viterbivars_t.append(best_tag) # 22 * 32

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # forward_var = torch.cat(viterbivars_t).view(self.batch_size, -1).to("cuda") + feat # 128*22
            # backpointers.append(torch.cat(bptrs_t).view(self.batch_size, -1)) # 60*128*22

            backpointers.append(torch.stack(bptrs_t, dim=1)) # 60 * (32, 22)
            forward_var = torch.stack(viterbivars_t, dim=1).to("cuda") + feat # (32, 22) 把60词的概率全部加在一起

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[:, self.tag_to_ix[STOP_TAG], :] # (32, 22)
        # best_tag_id = torch.argmax(terminal_var, dim=1) # 128
        # path_score = [terminal_var[i][best_tag_id[i]] for i in range(self.batch_size)] # 128
        path_score, best_tag_id = terminal_var.max(dim=1)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            # best_tag_id = [bptrs_t[i][best_tag_id[i]].cpu().item() for i in range(self.batch_size)] # 128
            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id.unsqueeze(dim=1)).squeeze(dim=1)
            best_path.append(best_tag_id) #61*128
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        # print(start.type())
        # ye = torch.tensor([self.tag_to_ix[START_TAG]])
        # print(ye.type())
        # assert start == torch.tensor([self.tag_to_ix[START_TAG]]).expand(self.batch_size).long().to("cuda")  # Sanity check
        best_path.reverse()
        # best_path = torch.tensor(best_path).view(self.batch_size, -1)
        best_path = torch.stack(best_path, dim=1)
        return path_score, best_path

    def forward(self, sentence):  # don't confuse this with _forward_alg under.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000.).to("cuda")
        # START_TAG has all of the score.
        for i in range(self.batch_size):
            init_alphas[i][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            # feat.shape = 32*22
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    self.batch_size, -1).expand(self.batch_size, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[:, next_tag, :].view(self.batch_size, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            # forward_var = torch.cat(alphas_t).view(self.batch_size, -1)
            forward_var = torch.stack(alphas_t, dim=1)
        terminal_var = forward_var + self.transitions[:, self.tag_to_ix[STOP_TAG], :]
        alpha = torch.logsumexp(terminal_var, dim=1) # [32, 22]
        return alpha #[32]

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(self.batch_size).to("cuda")
        # tags.shape = batch_size * seq_length
        # 给每一句话前面加上"<START>"
        # tags = torch.cat((torch.tensor([self.tag_to_ix[START_TAG]]*self.batch_size, dtype=torch.long).view(self.batch_size, -1).to("cuda"),
        #                   tags), dim=1)
        tags = torch.cat([torch.full([self.batch_size, 1], self.tag_to_ix[START_TAG], dtype=torch.long).to("cuda"), tags], dim=1)

        for i, feat in enumerate(feats):
            a = torch.tensor([self.transitions[j, tags[j, i + 1], tags[j, i]] for j in range(self.batch_size)], dtype=torch.float)
            # b = torch.tensor([feat[j, tags[j, i + 1]] for j in range(self.batch_size)], dtype=torch.float)
            b = torch.gather(feat, dim=1, index=tags[:, i+1].unsqueeze(dim=1)).squeeze(dim=1)
            score = score + \
                 a.to("cuda") + \
                    b.to("cuda")

        # score = score + torch.tensor([self.transitions[j, self.tag_to_ix[STOP_TAG], tags[j, -1]] for j in range(self.batch_size)], dtype=torch.float).to("cuda")
        score = score + torch.gather(self.transitions[:, self.tag_to_ix[STOP_TAG], :], dim=1, index=tags[:, -1].unsqueeze(dim=1)).squeeze(dim=1)
        return score


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence) #(60, 32, 22)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        # return torch.div((forward_score - gold_score).norm(1), self.batch_size)
        return (forward_score - gold_score).mean()


