from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import Phrases
import pandas as pd
import re

pd_title = pd.read_excel('./excel_sample/마버리타이틀.xlsx')
raw_title_list = pd_title.bd_title_ko.tolist()

raw_title_list

def refined_special_symbol(str_list):
    # 특수문자 제끼기
    return [re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', obj) for obj in str_list]

title_list = refined_special_symbol(raw_title_list)
corpus_list = [obj.split() for obj in title_list]
bigram_transformer = Phrases(corpus_list)

model = Word2Vec(bigram_transformer[corpus_list], size=2, window=1, min_count=0, workers=4, negative=5)
# size 가운데 동그라미 히든 3개
#
# model.save('./wordvec.model')
# model.hashfxn()

model.most_similar('바삭한')


model.wv.most_similar(positive=['페스티벌', '거리'], negative=['페스티벌'])
model.wv['해리포터']
model.wv.syn0.shape #weight

model.get_weights()
# word_vectors = model.wv
# word_vectors

# model = Word2Vec(bigram_transformer[common_texts], min_count=1)