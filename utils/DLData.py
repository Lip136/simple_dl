# encoding:utf-8
'''
功能：pytorch的数据管理模块
'''

import torch.utils.data as data

# 读取数据构建Dataset子类
class MyDataset(data.Dataset):
    def __init__(self, input_seqs, targets):
        self.input_seqs = input_seqs
        self.targets = targets

    def __getitem__(self, index):
        # 返回tensor
        x, y = self.input_seqs[index], self.targets[index]
        return x, y

    def __len__(self):
        return len(self.input_seqs)
# dataset = MyDataset(input_seqs, targets)

# 生成batch数据
# data.DataLoader(
# dataset,
# batch_size=1,
# shuffle=False,
# sampler=None,
# num_workers=0,
# collate_fn=<function default_collate>,
# pin_memory=False,
# drop_last=False)

