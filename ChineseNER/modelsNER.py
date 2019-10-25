# encoding:utf-8
import torch
import torch.nn as nn
import torchsnooper

torch.manual_seed(1)


class BiLSTM_CRF(nn.Module):
    def __init__(self, config, vocab):

        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = config["emb_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.tagset_size = config["tagset_size"]

        self.embedding = nn.Embedding(vocab.size(), self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        t = torch.randn(self.tagset_size, self.tagset_size)
        self.transitions = nn.Parameter(nn.init.kaiming_normal_(t))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.START_TAG = vocab.get_id("<start>")
        self.STOP_TAG = vocab.get_id("<end>")
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000


    # @torchsnooper.snoop()
    def _get_lstm_features(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats # (seq_len, batch_size, tagset_size)

    # @torchsnooper.snoop()
    def _viterbi_decode(self, feats):

        backpointers = []
        batch_size = feats.size(1)

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((batch_size, self.tagset_size), -10000.).to("cuda")
        init_vvars[:, self.START_TAG] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:

            next_tag = forward_var.unsqueeze(dim=1) + self.transitions.unsqueeze(dim=0)
            viterbivars_t, bptrs_t = next_tag.max(dim=2) # (batch_size, tagset_size)

            backpointers.append(bptrs_t)
            forward_var = viterbivars_t + feat

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG].unsqueeze(dim=0)

        path_score, best_tag_id = terminal_var.max(dim=1) # (batch_size)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):

            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id.unsqueeze(dim=1)).squeeze(dim=1)
            best_path.append(best_tag_id) # (seq_len, batch_size)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        # assert start == torch.tensor([self.START_TAG] * batch_size).long()  # Sanity check
        best_path.reverse()

        best_path = torch.stack(best_path, dim=1) # 这里相当于做了一个转置
        return path_score, best_path # (batch_size), (batch_size, seq_len)

    def forward(self, sentence):  # don't confuse this with _forward_alg under.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


    def _forward_alg(self, feats):
        batch_size = feats.size(1)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            next_tag = forward_var.unsqueeze(dim=1) + feat.unsqueeze(dim=1) + self.transitions.unsqueeze(dim=0)
            forward_var = torch.logsumexp(next_tag, dim=2)

        terminal_var = forward_var + self.transitions[self.STOP_TAG].unsqueeze(dim=0) # (batch_size, tagset_size)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        batch_size = feats.size(1)
        tags = torch.cat([torch.full([batch_size, 1], self.START_TAG, dtype=torch.long).to("cuda"), tags], dim=1)

        tran = []
        for j in range(batch_size):
            tran.append(self.transitions[tags[j, 1:], tags[j, :-1]].sum())

        feat_socre = torch.gather(feats.transpose(0, 1), dim=2, index=tags[:, 1:].unsqueeze(dim=2)).squeeze(dim=2).sum(dim=1)
        score = torch.stack(tran, dim=0) + feat_socre

        score = score + torch.gather(self.transitions[self.STOP_TAG, :].expand(batch_size, self.tagset_size), dim=1, index=tags[:, -1].unsqueeze(dim=1)).squeeze(dim=1)
        return score


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence) #(60, 32, 22)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()




