from os import listdir
from os.path import isfile, join
import numpy as np
import collections
import sys


def prepare_data(data_dir):
    absolute_dir = join(sys.path[0], data_dir)
    files = [file for file in listdir(absolute_dir) if isfile(join(absolute_dir, file))]
    sentences = []
    for file in files:
        with open(join(absolute_dir, file)) as f:
            sentences.extend(map(lambda x: x.lower(), f.read().split()))
    return sentences


def build_vocabulary(words):
    word_counts = collections.Counter(words)
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def create_batches(vocabulary, text, batch_size, seq_length):
    tensor = np.array(list(map(vocabulary.get, text)))
    num_batches = int(tensor.size / (batch_size * seq_length))
    tensor = tensor[:num_batches * batch_size * seq_length]
    x_data = tensor
    y_data = np.copy(tensor)
    y_data[:-1] = x_data[1:]
    y_data[-1] = x_data[0]
    x_batches = np.split(x_data.reshape(batch_size, -1), num_batches, 1)
    y_batches = np.split(y_data.reshape(batch_size, -1), num_batches, 1)
    return [x_batches, y_batches]
