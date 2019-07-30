# encoding:utf-8
import chardet
# 批量处理文件
class file(object):
    def __init__(self, file_path):
        self.file_path = file_path

    # 获取文件编码类型
    def get_encoding(self):
        with open(self.file_path, 'rb') as f:
            return chardet.detect(f.read())['encoding']


