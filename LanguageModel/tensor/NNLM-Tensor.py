# encoding:utf-8
import tensorflow as tf
import data_options
import numpy as np


# tf.reset_default_graph()

sentences = ["嫦娥四号 任务 圆满 成功", "开启了 人类 探索 宇宙奥秘", "月球背面 软着陆 和 巡视勘察"]
# import jieba
# for i in range(len(sentences)):
#     sentences[i] = " ".join(list(jieba.cut(sentences[i])))
# 这个时候就需要加入padding
embed = data_options.embed_word(sentences)
word2n, number2w, n_class = embed.word_dict()
input_batch, target_batch = embed.get_batch()

class NNLM(object):
    '''
    功能： 语言模型
    输入：embed的数据和标签
    输出：模型
    '''
    def __init__(self):
        self.n_step = 3 # 输入前三个词
        self.hidden_size = 2
        self.epoch = 5000

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

        # save info

    def _build_graph(self):
        self._setup_placeholder()
        self._train_epoch()

    def _setup_placeholder(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
        self.Y = tf.placeholder(tf.float32, [None, n_class])
        self.H = tf.Variable(tf.random_normal([self.n_step * n_class, self.hidden_size]))
        self.W = tf.Variable(tf.random_normal([self.n_step * n_class, n_class]))
        self.d = tf.Variable(tf.random_normal([self.hidden_size]))
        self.U = tf.Variable(tf.random_normal([self.hidden_size, n_class]))
        self.b = tf.Variable(tf.random_normal([n_class]))


    def _train_epoch(self):

        input = tf.reshape(self.X, shape=[-1, self.n_step * n_class])  # [batch_size, n_step * n_class]
        tanh = tf.nn.tanh(self.d + tf.matmul(input, self.H))  # [batch_size, n_hidden]
        model = tf.matmul(input, self.W) + tf.matmul(tanh, self.U) + self.b  # [batch_size, n_class]

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.prediction = tf.argmax(model, 1)

    def train(self):
        # Training

        for epoch in range(self.epoch):
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_batch, self.Y: target_batch})
            if (epoch + 1)%1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    def predict(self):

        # Predict预测也是拿的input_batch数据
        predict = self.sess.run([self.prediction], feed_dict={self.X: input_batch})
        predict = [number2w[tuple(np.eye(n_class)[n].tolist())] for n in predict[0]]

        return predict


model = NNLM()
model.train()
# Test
print(model.predict())