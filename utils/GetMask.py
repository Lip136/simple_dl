# encoding:utf-8

import torch
import numpy as np


def nopeak_mask(seq_len, device=None):
    # 上三角
    np_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype("uint8")
    # 下三角
    np_mask = torch.from_numpy(np_mask) == 0

    if device:
        np_mask = np_mask.to(device)
    return np_mask

def create_masks(source, target, config):
    """
    params: source.shape = (batch_size, src_len)
            target.shape = (batch_size, trg_len)
            config is dictionary
    return: src_mask.shape = (batch_size, 1, src_len)
            trg_mask.shape = (batch_size, trg_len, trg_len)
    """
    src_mask = (source != config["src_pad"]).unsqueeze(dim=1)
    if target:
        trg_mask = (target != config["trg_pad"]).unsqueeze(dim=1)
        seq_len = target.size(1)
        np_mask = nopeak_mask(seq_len, config["device"])
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

