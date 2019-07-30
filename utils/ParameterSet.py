# encoding:utf-8
'''
功能：一个可以引入的parameter集合
'''
import argparse

def ParaSet():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', default='./data/english.txt')
    parser.add_argument('-trg_data', default='./data/french.txt')
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='fr') #required=True,
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights', default='weights/')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=200)
    parser.add_argument('-checkpoint', type=int, default=15)

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = ParaSet()
    opt.device = 0 # 单独加上参数
    print(opt)