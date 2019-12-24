import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import print_tensor
import numpy as np
import pandas as pd
import re
import sys

pd_title = pd.read_excel('./excel_sample/마버리타이틀.xlsx')
raw_title_list = pd_title.bd_title_ko.tolist()

def refined_special_symbol(str_list):
    # 특수문자 제끼기
    return [re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', obj) for obj in str_list]


title_list = refined_special_symbol(raw_title_list)

corpus = title_list[0]

# 전처리 해야함
le = LabelEncoder()
enc = OneHotEncoder()

corpus_list = corpus.split()

le_list=le.fit_transform(corpus_list).reshape(len(corpus_list), -1)

le.inverse_transform(le_list)
le_list

enc.fit(le_list)
corpus_oh = enc.transform(le_list).toarray().tolist()
contexts = []
target = []

window_size = 1
for i, co in enumerate(corpus_oh):
    if len(corpus_oh) - 1 == i:
        break
    if i < window_size:
        continue
    contexts.append(co)
    target.append([corpus_oh[i-1], corpus_oh[i+1]])

contexts = np.array(contexts)
target = np.array(target)
#모델 만들기

target=tf.transpose(a=target, perm=[1,0,2])

input_shape = (contexts.shape[-1])
dimensoin = 3

input = Input(shape=input_shape)  # input 2개중에 1개가 행렬 1x7로 이루어져있기 때문에 7이 들어감
x = Dense(3, name='pre')(input)   # matmul 부분 Dense는 처음 weight rand까지 다

# dot = tf.add(x1, x2) * 0.5  # 책에있는 공식이다
dot = x

outs1 = Dense(input_shape, activation='softmax', name='output1')(dot)  # Weight x dot을 한후 소프트 맥스까지 한다.
outs2 = Dense(input_shape, activation='softmax', name='output2')(dot)

skipgram = Model(inputs=input, outputs=[outs1, outs2])

skipgram.summary()


skipgram.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #adam = 통용적으로쓴다 metrics categorical_crossentropy 다수 binary_엔트로피는 2개할때

skipgram.fit(
    contexts, #입력
    [target[0],target[1]], # 정답
    batch_size=3,# input 3덩어리 씩 입력하여 진행 하겠다.
    epochs=1000 # 몇바퀴
#     verbose=1
)

np.save('./skipgram_emb.npy', skipgram.get_weights()[0])

w_in = skipgram.get_weights()[0]

np.array(contexts)
target_pre = skipgram.predict(contexts)


target_pre = np.array(target_pre)
target = np.array(target)

target_pre.argmax(axis=1)
target.argmax(axis=1)

def get_label(y_target, y_pred):
    y_true, pred = y_target.argmax(axis=1), y_pred.argmax(axis=1)

    return np.mean(pred == y_true)

le_list
corpus_list
corpus_oh
ee = target_pre.argmax(axis=1)
get_label(target, target_pre)
corpus_list

le.inverse_transform(contexts.argmax(axis=1))
le.inverse_transform(ee[0])
le.inverse_transform(ee[1])