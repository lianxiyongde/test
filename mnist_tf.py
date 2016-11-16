import tensorflow as tf
import numpy as np
import pandas as pd
import os


os.chdir('/Desktop/mnist_data')
df = pd.read_csv('digit_recognition_train.csv')

trX = np.array([df.iloc[i][1:]/255 for i in range(32000)])
teX = np.array([df.iloc[i][1:]/255 for i in range(32000, 42000)])

trY = np.zeros((32000, 10))
for i in range(32000):
    trY[i][int(df['label'][i])] = 1.0

teY = np.zeros((10000, 10))
for i in range(10000):
    teY[i][int(df['label'][i+32000])] = 1.0

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = tf.Variable(tf.random_normal([784, 625], stddev=0.01))
w_o = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

h = tf.nn.relu(tf.matmul(X, w_h))
y_hat = tf.matmul(h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, Y)) 
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) 
test_op = tf.argmax(y_hat, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(30):
        for start, end in zip(range(0, len(trX), 100), range(100, len(trX)+1, 100)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(test_op, feed_dict={X: teX})))

sess.close()
