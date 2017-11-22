BATCH_SIZE = 50
SEQ_LENGTH = 25

from lighthouse.prepare import create_batches, prepare_data, build_vocabulary
from lighthouse.model import train


def run_training(data_dir):
    words = prepare_data(data_dir)
    vocabulary = build_vocabulary(words)
    batches = create_batches(vocabulary, words, BATCH_SIZE, SEQ_LENGTH)
    print('text length: {}, vocabulary length: {}, batches count: {}'.format(len(words), len(vocabulary), len(batches[0])))
    train(BATCH_SIZE, SEQ_LENGTH, len(vocabulary), batches[0], batches[1])

run_training('..\data\lovecraft')

