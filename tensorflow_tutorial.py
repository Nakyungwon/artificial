import tensorflow as tf
import math
import numpy as np
from tensorflow import keras

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# W = tf.Variable(2.0, name='weight')
# b = tf.Variable(0.7, name='bias')
#
# for x in [1.0, 0.6, -1.8]:
#     z = W * x + b
#     print('x = %4.1f --> z = %4.1f' % (x, z))

# 순전파
X = tf.Variable([[1.0, 2.0]], name='X') # 입력값
W1 = tf.Variable(tf.random.uniform(shape=(2,3), minval=1, maxval=5, name="W")) # 가중치1
b1 = tf.Variable(tf.ones(shape=(3)), name="bias") # 편향1

m1 = tf.matmul(X, W1) + b1 # 곱하고 평향
h1 = tf.nn.sigmoid(m1) # 시그모이드
h1

W2 = tf.Variable(tf.random.uniform(shape=(3,3), minval=1, maxval=5, name="W"))
b2 = tf.Variable(tf.ones(shape=(3)), name="bias")

m2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.sigmoid(m2)


def softmax(h2):
    rows = h2.shape[1]
    total = 0
    tmp = []
    for row in range(0, rows):
        total += tf.math.exp(h2[0][row])
    print(total)
    for row in range(0, rows):
        tmp.append(tf.math.exp(h2[0][row]) / total)
    return tf.Variable([tmp])


Y = softmax(h2) # 소프트 맥스
T = tf.Variable([[0 ,0 ,1]], dtype=tf.float32) # 3번째가 정답

def cross_entropy(T, Y):
    # T 와 Y의 차원이 같아야함
    rtn = - tf.reduce_sum(tf.multiply(T, tf.math.log(Y)))
    return rtn

L = cross_entropy(T, Y)
L

#역전파
Y - T
Y
T
def backword(Y, T):
    return Y - T
dh2 = backword(Y, T)

h1
tf.nn.sigmoid(tf.matmul(h1, W2))
dW2 = h1 * tf.nn.sigmoid(tf.matmul(h1, W2)) * (1 - tf.nn.sigmoid(tf.matmul(h1, W2))) * dh2
dh1 = W2 * tf.nn.sigmoid(tf.matmul(h1, W2)) * (1 - tf.nn.sigmoid(tf.matmul(h1, W2))) * dh2 # matmul???
dW1 = X * tf.nn.sigmoid(tf.matmul(X, W1)) * (1 - tf.nn.sigmoid(tf.matmul(X, W1))) * dh1
