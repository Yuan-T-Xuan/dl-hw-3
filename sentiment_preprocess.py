import re
import pickle
import numpy as np

def create_vocab(sourcefile_path, output_path):
    f = open(sourcefile_path)
    lines = f.read().split("\n")[:-1]
    f.close()
    vocab = dict()
    for line in lines:
        alpha_only = re.sub("[^a-z ]", "", line.lower())
        for word in alpha_only.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    print(vocab)
    outf = open(output_path, 'wb')
    pickle.dump(vocab, outf)
    outf.close()

def sentence2vector(sentence_str, vocab, maxlen=400):
    sentence_str = re.sub("[^a-z ]", "", sentence_str.lower())
    words = sentence_str.split()
    words = [vocab[w] for w in words]
    if len(words) > maxlen:
        words = words[:maxlen]
    elif len(words) < maxlen:
        words = [0] * (maxlen - len(words)) + words
    return words

def sentences2matrix(file_path, vocab, maxlen=400):
    f = open(file_path)
    lines = f.read().split("\n")[:-1]
    f.close()
    matrix = []
    for line in lines:
        matrix.append(sentence2vector(line, vocab, maxlen))
    return np.array(matrix, dtype=int)

"""
from sentiment_preprocess import *
import pickle
import numpy as np
import torch
from torch.utils.data import *

f = open("data/vocab.pickle", 'rb')
vocab = pickle.load(f)
m = sentences2matrix("data/train_pos_merged.txt", vocab)
train_data = TensorDataset(torch.from_numpy(m), torch.from_numpy(np.ones(1500, dtype=int)))
"""
