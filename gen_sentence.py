#encoding: utf-8
#
# Copyright (c) 2016 chainer_nlp_man
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
import argparse
import math
import sys
import itertools
import random
import bisect

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

import net

# 引数にモデルファイルを指定
parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='',
                    help='the model from given file')
args = parser.parse_args()

# 単語<->ID変換用
vocab2id = {}
id2vocab = {}

# train_ptb.pyと同じ読み込み方にすることで単語とIDのペアが一致するようにする
def load_data(filename):
    global vocab2id, id2vocab, n_vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    for i, word in enumerate(words):
        if word not in vocab2id:
            vocab2id[word] = len(vocab2id)
            id2vocab[vocab2id[word]] = word

    dataset = np.ndarray((len(words),), dtype=np.int32)
load_data('ptb.train.txt')
load_data('ptb.valid.txt')
load_data('ptb.test.txt')

# train_ptb.pyと同じ設定にする
n_units = 650

lm = net.RNNLM(len(vocab2id), n_units, False)
model = L.Classifier(lm)

# モデルデータの読み込み
serializers.load_hdf5(args.model, model)

# 文の適当な生成
for i in range(0,1):
    print(i+1, end=": ")
    # モデルの状態をいったんリセット
    model.predictor.reset_state()
    word = "<eos>"
    while True:
        # RNNLMへの入力を準備
        x = chainer.Variable(np.array([vocab2id[word]],dtype=np.int32))
        # RNNLMの出力のsoftmaxを取得
        y = F.softmax(model.predictor(x))
        print(y.data[0][vocab2id["for"]])
        # 各単語の確率値として、単語をサンプリングし、次の単語とする
        y_accum = np.add.accumulate(y.data[0])
        #print(y_accum[-1])
        #print(y.data[0])
        r = random.random()
        word = id2vocab[bisect.bisect(y_accum, r)]
        # もし文末だったら終了
        if word == "<eos>":
            print(".")
            break
        else:
            print(word, end=" ")
