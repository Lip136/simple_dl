# encoding:utf-8

import codecs

# x=sentence (32, 60), y=tags (32, 60)
def calculate(x, y, id2word, id2tag):

    # 每一个batch里面，每一个句子里的词属于什么，其实我们是根据y先把label取出来，然后再去取x中的词语
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    batch_entity = []
    for i in range(len(x)):
        # i是每一个句子
        sen_entity = []
        entity = []
        for j in range(len(x[i])):

            # 每句子下的词
            if x[i][j] == 0 or y[i][j] == 0: # word=" ", tag=" "
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                sen_entity.append(entity)
                entity = []
            else:
                entity = []
        batch_entity.append(sen_entity)

    return batch_entity


    # for j in range(len(x)):
    #     if x[j] == 0 or y[j] == 0:
    #         continue
    #     if id2tag[y[j]][0] == 'B':
    #         entity = [id2word[x[j]] + '/' + id2tag[y[j]]]
    #     elif id2tag[y[j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
    #         entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
    #     elif id2tag[y[j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
    #         entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
    #         entity.append(str(j))
    #         res.append(entity)
    #         entity = []
    #     else:
    #         entity = []
    # return res


def calculate3(x, y, id2word, id2tag, res=[]):
    '''
    使用这个函数可以把抽取出的实体写到res.txt文件中，供我们查看。
    注意，这个函数每次使用是在文档的最后添加新信息，所以使用时尽量删除res文件后使用。
    '''
    with codecs.open('./res.txt', 'a', 'utf-8') as outp:
        entity = []
        for j in range(len(x)):  # for every word
            if x[j] == 0 or y[j] == 0:
                continue
            if id2tag[y[j]][0] == 'B':
                entity = [id2word[x[j]] + '/' + id2tag[y[j]]]
            elif id2tag[y[j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
                entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
            elif id2tag[y[j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
                entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
                entity.append(str(j))
                res.append(entity)
                st = ""
                for s in entity:
                    st += s + ' '
                # print st
                outp.write(st + '\n')
                entity = []
            else:
                entity = []
    return res
