import numpy as np
import random
import math

from pipeline.tokenizer import PyTokenizer
from utils import read_file

DEBUG = False


class DataLoader:
    def __init__(self, tokenizer, train_percent=.9, sample_len=50, step=1, one_hot_input=False,
                 data_file='../data/clean.py', shuffle_samples=True):
        self.tokenizer = tokenizer
        self.train_percent = train_percent
        self.sample_len = sample_len
        self.step = step
        self.one_hot_input = one_hot_input
        if self.one_hot_input:
            raise NotImplementedError('One hot for input sequences not implemented')
        self.data_file = data_file
        self.shuffle_samples = shuffle_samples

        self.statements = []
        self.next_statements = []
        self.init_statements()

    def init_statements(self):
        data = read_file(self.data_file)
        if self.tokenizer.real_vocab_len == 0:
            self.tokenizer.fit_on_data(data)

        tokens = self.tokenizer.text_to_sequence(data)

        for i in range(0, len(tokens) - self.sample_len, self.step):
            self.statements.append(tokens[i:i + self.sample_len])
            self.next_statements.append(tokens[i + self.sample_len])

        if self.shuffle_samples:
            zipped = list(zip(self.statements, self.next_statements))
            random.shuffle(zipped)
            self.statements, self.next_statements = zip(*zipped)

        if DEBUG:
            print('total number of samples:', len(self.statements))

    def get_training_data(self):
        split_pt = math.floor(len(self.statements) * self.train_percent)
        x = np.array(self.statements[:split_pt], dtype=np.int)
        y = np.zeros((split_pt, self.tokenizer.real_vocab_len), dtype=np.int)
        for i, next_statement in enumerate(self.next_statements[:split_pt]):
            y[i, next_statement] = 1

        return x, y

    def get_test_data(self):
        split_pt = math.floor(len(self.statements) * self.train_percent)
        x = np.array(self.statements[split_pt:], dtype=np.int)
        y = np.zeros((split_pt, self.tokenizer.real_vocab_len), dtype=np.int)
        for i, next_statement in enumerate(self.next_statements[split_pt:]):
            y[i, next_statement] = 1

        return x, y


if __name__ == '__main__':
    t = PyTokenizer(5000)

    dl = DataLoader(t)
    train_x, train_y = dl.get_training_data()
    test_x, test_y = dl.get_test_data()