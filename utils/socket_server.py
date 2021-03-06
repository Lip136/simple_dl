# encoding:utf-8
'''
功能：socket连接
1.发送到服务器
2.接受服务器的数据进行处理
'''
import socket

class socket_text(object):
    def __init__(self, server_ip, server_port):
        self.sk = socket.socket()
        self.server_ip = server_ip
        self.server_port = server_port
    # 发送到服务器
    def send(self, text):
        self.sk.connect((self.server_ip, self.server_port))
        self.sk.send(text)
        from socket import error as SocketError
        import errno
        try:
            ret = self.sk.recv(2048)
            print(ret.decode('utf-8'))
        except SocketError as e:
            if e.errno != errno.ECONNRESET:
                raise
            pass
        self.sk.close()
    # 接受服务器的数据进行处理
    def accept(self):
        self.sk.bind((self.server_ip,self.server_port))
        self.sk.listen(5)
        conn, addr = self.sk.accept()
        ret = conn.recv(1024)
        print(ret)
        conn.send(b'hi')
        conn.close()
        self.sk.close()

if __name__ == "__main__":

    # questions = ["baby是什么意思", "baby有什么词组和句子", "baby和boy可以组成什么短语", "mother和baby可以组成什么句子"]
    # for question in questions:
    #     sk = socket_text('10.3.10.194', 8089)
    #     sk.send(question.encode('utf-8'))
    sk = socket_text('10.3.27.36', 8086)
    sk.send("周星驰演过什么电影".encode('utf-8'))


