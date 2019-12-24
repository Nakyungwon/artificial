from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from konlpy.tag import Kkma
from konlpy.utils import pprint
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import Phrases

# model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# model.add(layers.Embedding(input_dim=len(corpus_list), output_dim=4))

# Add a LSTM layer with 128 internal units.
# model.add(layers.LSTM(128))

# Add a Dense layer with 10 units and softmax activation.
# model.add(layers.Dense(4))
#
# model.add(layers.Embedding(input_dim=len(corpus_list), output_dim=4))

text = "<start> 나는 오늘도 AI를 공부한다 왜냐하면 내가사는 세상은 AI로 가득차 있기 때문이다 <end>"
corpus_list = text.split()
[corpus_list]
corpus_list

word_2_vec_model = Word2Vec([corpus_list], size=4, window=1, min_count=0, workers=4, negative=2)
word_2_vec_model.wv.syn0[0]

text_as_int = np.array([idx for idx, obj in enumerate(corpus_list)])
text_as_int

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
char_dataset


sequences = char_dataset.batch(2, drop_remainder=True)

char2idx = {u:i for i, u in enumerate(corpus_list)}
char2idx.values()
idx2char = np.array(corpus_list)
idx2char

char_dataset.take(10)
for i in char_dataset.take(10):
  print(char_dataset.take(10))
  print(i)
  print(i.numpy())
  print(corpus_list[i.numpy()])

def split_input_target(chunk):
    print(chunk)
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

char_dataset
dataset = sequences.map(split_input_target)
dataset

vocab_size = len(corpus_list)
word_2_vec_model['<start>']
word_2_vec_model.most_similar('왜냐하면')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 4,
                              batch_input_shape=[1, None]),
    tf.keras.layers.LSTM(1024,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])

model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

history = model.fit(dataset, epochs=10)

target_pre=model.predict(text_as_int)

target_pre.argmax(axis=1)

def accuracy_euni(y_target, y_pred):
    y_true, pred = y_target.argmax(axis=1), y_pred.argmax(axis=1)

    return np.mean(pred == y_true)



char2idx.values()
# '나는' -> 모델 -> '오늘도' 어떻게??
# model.sdlfldsf('나는')
