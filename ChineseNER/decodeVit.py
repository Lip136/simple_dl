# encoding:utf-8
"""
Viterbi解码
"""
import torch
import torchsnooper
# @torchsnooper.snoop()
def viterbi_decode(vocab, feats):
    """
    输入发射概率
    :param feats: shape = [seq_len, batch_size, class]
    :return: 
    """
    backpointers = []
    batch_size = feats.size(1)
    tagset_size = feats.size(2)
    # Initialize the viterbi variables in log space
    # 初始状态为何全为-10000?
    init_vvars = torch.full((batch_size, tagset_size), -10000.).to("cuda")

    init_vvars[:, vocab.get_id("<start>")] = 0.

    transitions = torch.randn([tagset_size, tagset_size]).to("cuda")
    transitions[vocab.get_id("<start>"), :] = -10000.
    transitions[:, vocab.get_id("<end>")] = -10000.
    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = init_vvars
    for feat in feats:

        next_tag = forward_var.unsqueeze(dim=1) + transitions.unsqueeze(dim=0)
        viterbivars_t, bptrs_t = next_tag.max(dim=2)  # (batch_size, tagset_size)

        backpointers.append(bptrs_t)
        forward_var = viterbivars_t + feat


    # Transition to STOP_TAG
    terminal_var = forward_var + transitions[vocab.get_id("<end>")].unsqueeze(dim=0)  # (32, 22)
    path_score, best_tag_id = terminal_var.max(dim=1)

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        # best_tag_id = [bptrs_t[i][best_tag_id[i]].cpu().item() for i in range(self.batch_size)] # 128
        best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id.unsqueeze(dim=1)).squeeze(dim=1)
        best_path.append(best_tag_id)  # 61*128
    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    point = torch.tensor([vocab.get_id("<start>")] * batch_size).long().to("cuda")
    assert sum(start == point) == batch_size
    best_path.reverse()

    best_path = torch.stack(best_path, dim=1)

    return path_score, best_path
class ner():
    def __init__(self):
        self.START_TAG = 0
        self.STOP_TAG = 1
        self.tagset_size = 22
        self.transitions = torch.randn(self.tagset_size, self.tagset_size).to("cuda")

    def _forward_alg(self, feats):
        batch_size = feats.size(1)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.).to("cuda")
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
        print(alpha.shape)
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
        print(score.shape)
        return score

class Vocab():
    def __init__(self):
        self.word2id = {"<start>":1, "<end>":2}

    def get_id(self, token):
        return self.word2id[token]

vocab = Vocab()
feats = torch.randn([200, 64, 22]).to("cuda")
tags = torch.randint(low=2, high=22, size=(64, 200)).to("cuda")
viterbi_decode(vocab, feats)


ne = ner()
x = ne._forward_alg(feats) - ne._score_sentence(feats, tags)
print(x.mean())

