# encoding:utf-8
'''
功能：pytorch的数据管理模块
'''

import torch.utils.data as data
import os, csv, glob, random
import torch
import pandas as pd

# 读取数据构建Dataset子类
class NERDataset(data.Dataset):
    def __init__(self, root,  mode, max_length):
        self.root = root
        self.word2id = {" ": 0}
        self.label2id = {" ": 0}
        self.max_length = max_length

        self.seqs, self.labels = self.load_tsv()

        self.label2id["<START>"] = len(self.label2id)
        self.label2id["<STOP>"] = len(self.label2id)

        if mode == 'train': # 60%
            self.seqs = self.seqs[:int(0.6 * len(self.seqs))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val': # 20% = 60% -> 80%
            self.seqs = self.seqs[int(0.6 * len(self.seqs)):int(0.8 * len(self.seqs))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # 20% = 80% -> 100%
            self.seqs = self.seqs[int(0.8 * len(self.seqs)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_tsv(self):
        if not os.path.exists(self.root):
            print("原始文件不存在")
        else:
            seqs, labels = [], []
            seq, label = [], []
            with open(os.path.join(self.root), mode='r') as f:
                for line in f.readlines():
                    if line == '\n':
                        assert len(seq) == len(label)
                        seqs.append(seq)
                        labels.append(label)
                        seq = []
                        label = []
                    else:
                        word2label = line.split()
                        if len(word2label) == 2:
                            w_tmp, l_tmp = word2label[0], word2label[1]
                        elif len(word2label) == 1:
                            w_tmp, l_tmp = ' ', word2label[0]

                        if w_tmp not in self.word2id.keys():
                            self.word2id[w_tmp] = len(self.word2id)
                        if l_tmp not in self.label2id.keys():
                            self.label2id[l_tmp] = len(self.label2id)
                        seq.append(self.word2id[w_tmp])
                        label.append(self.label2id[l_tmp])


            assert len(seqs) == len(labels)
            return seqs, labels


    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        # 对每一个seq和label进行处理：padding
        seq, label = self.seqs[index], self.labels[index]
        assert len(seq) == len(label)
        max_len = self.max_length
        def padding(ids):
            if len(ids) >= max_len:
                return ids[:max_len]
            ids.extend([0] * (max_len - len(ids)))
            return ids

        seq = padding(seq)
        label = padding(seq)

        return seq, label

def main():
    root = './data/boson/Bonsondata.tsv'
    db = NERDataset(root, mode="val", max_length=60)
    print(db.label2id, len(db.word2id))

    data_load = data.DataLoader(db, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    print(next(iter(data_load)))

if __name__ == "__main__":
    main()




