import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

corpus='you say goodbye and i say hello .'
# contexts, target = conver_one_hot(corpus, 1)
window_size = 1

le = LabelEncoder()
enc = OneHotEncoder()


corpus_list = corpus.split()
corpus_list
# tf.one_hot(len(corpus_list), len(corpus_list))
# tf.Tran

le_fit = le.fit_transform(corpus_list)
le_fit
le_list = le_fit.reshape(len(corpus_list), -1) # -1 뒤에 알맞는 차원을 넣어줌
le_list
le_list.shape
enc.fit(le_list)
enc

corpus_oh = enc.transform(le_list).toarray().tolist()
corpus_oh

contexts = []
target = []
for i, co in enumerate(corpus_oh):
  if (i < window_size) or (i >= (len(corpus_oh) - window_size)):
    continue

  contexts.append([corpus_oh[i - window_size], corpus_oh[i + window_size]])
  print(contexts)
  target.append(co)

contexts = np.array(contexts)
contexts # 학습 데이터
target = np.array(target)
target # 목표데이터

contexts.shape

######################################################################
input_shape = (7,7)

from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Reshape
from tensorflow.keras.models import Model


inputs1 = Input(shape=input_shape[0])
inputs1
inputs2 = Input(shape=input_shape[1])
inputs2

x1 = Dense(3, name='pre')(inputs1)  # matmul 부분
x2 = Dense(3, name='post')(inputs2)

dot = tf.add(x1, x2) * 0.5

out = Dense(7, activation='softmax', name='output')(dot)

skipgram = Model(inputs=[inputs1, inputs2], outputs=out)



CBOW = Word2Vec_classification(input_shape)
CBOW.summary()
