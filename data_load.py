# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
By fangshuming519@gmail.com. 
https://www.github.com/FonzieTree/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
from num2words import num2words

def text_normalize(sent):
    '''Minimum text preprocessing'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = []
    for word in sent.split():
        word = _strip_accents(word.lower())
        srch = re.match("\d[\d,.]*$", word)
        if srch:
            word = num2words(float(word.replace(",", "")))
        word = re.sub(u"[-—-]", " ", word)
        word = re.sub("[^ a-z'.?]", "", word)
        normalized.append(word)
    normalized = " ".join(normalized)
    normalized = re.sub("[ ]{2,}", " ", normalized)
    normalized = normalized.strip()

    return normalized

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_data(training=True):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, mels, dones, mags = [], [], [], []
    num_samples = 1
    metadata = os.path.join(hp.data, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, sent = line.strip().split("|")
        sent = text_normalize(sent) + "E" # text normalization, E: EOS
        if len(sent)<=hp.Tx:
            texts.append([char2idx[char] for char in sent] + [0] * (hp.Tx - len(sent)))
        else:
            texts.append([char2idx[char] for char in sent][:hp.Tx])
        mels.append(os.path.join(hp.data, "mels", fname + ".npy"))
        dones.append(os.path.join(hp.data, "dones", fname + ".npy"))
        mags.append(os.path.join(hp.data, "mags", fname + ".npy"))
    return texts, mels, dones, mags


def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
        if len(sent) <= hp.Tx:
            sent += "P"*(hp.Tx-len(sent))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _mels, _dones, _mags = load_data()
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts[:hp.batch_size], tf.int32)
        mels = tf.convert_to_tensor(np.array([np.load(_mels[i]) for i in range(hp.batch_size)]), tf.float32)
        dones = tf.convert_to_tensor(np.array([np.load(_dones[i]) for i in range(hp.batch_size)]), tf.int32)
        mags = tf.convert_to_tensor(np.array([np.load(_mags[i]) for i in range(hp.batch_size)]), tf.float32)

    return texts, mels, dones, mags
