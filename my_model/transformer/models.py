# encoding:utf-8
'''
功能：transformer的模型
参数：

'''

import torch
import copy, os
import torch.nn as nn

from Embedder import EmbedBase, PositionalEncoder
from Layer import EncoderLayer, DecoderLayer
from SubLayer import Norm

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.N = config["encoder_num"]
        self.emb_dim = config["emb_dim"]

        self.embed = EmbedBase(config["source_size"], self.emb_dim)
        self.pe = PositionalEncoder(self.emb_dim)
        self.layers = get_clones(EncoderLayer(config), self.N)
        self.norm =Norm(self.emb_dim)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.N = config["encoder_num"]
        self.emb_dim = config["emb_dim"]

        self.embed = EmbedBase(config["target_size"], self.emb_dim)
        self.pe = PositionalEncoder(self.emb_dim)
        self.layers = get_clones(DecoderLayer(config), self.N)
        self.norm = Norm(self.emb_dim)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

# 重头戏
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.emb_dim = config["emb_dim"]
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.out = nn.Linear(self.emb_dim, config["target_size"])

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_outputs)
        return output

# we don't perform softmax on the output as this will be handled
# automatically by our loss function

def get_model(config):
    assert config["emb_dim"] % config["heads"] == 0
    assert config["dropout"] < 1

    model = Transformer(config)
    pretrained = os.path.join(config["model_path"], config["model_data"])
    if os.path.exists(pretrained):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(pretrained))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if config["device"] == "GPU":
        model = model.cuda()

    return model

if __name__ == "__main__":
    config = {
        "source_size": 13725,
        "target_size": 23472,
        "encoder_num": 6,
        "decoder_num": 6,
        "emb_dim": 512,
        "heads": 8,
        "dropout": 0.1,
        "model_path": "../weights",
        "model_data": "model_weights",
        "device": "GPU"

    }
    model = get_model(config)
    print(sum([param.nelement() for param in model.parameters()]))

