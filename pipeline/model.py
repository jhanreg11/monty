import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class Model:
    def __init__(self, vocab_len, sample_len, **hyper_params):
        self.vocab_len = vocab_len
        self.sample_len = sample_len

        self.model = Sequential()
        self.model.add(Embedding(vocab_len, 32))
        self.model.add(LSTM(300, return_sequences=True))
        self.model.add(LSTM(300, return_sequences=True))
        self.model.add(LSTM(128))
        self.model.add(Dense(vocab_len, activation='softmax'))

        optimizer = RMSprop(lr=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def train(self, x, y, epochs=1, mini_batch_size=128):
        """
        train model on given data
        :param x: np.array, training inputs, dims (#samples, sample_len)
        :param y: np.array, training labels, dims (#samples, vocab_len)
        :param epochs: int, number of iterations to train on data
        :param mini_batch_size: int, size of mini batches
        :return: None
        """
        assert x.shape[1] == self.sample_len, 'Incorrect sample length. Given: {}, Expecting: {}'.format(
            x.shape[1], self.sample_len)
        self.model.fit(x, y, mini_batch_size, epochs)

    @staticmethod
    def sample_next_token(seed, temp):
        """
        sample next token given model's output sequence.
        :param seed: np.array, output from model used to pick next token
        :param temp: float, amount of randomness to use when sampling next token
        :return: int, index of sampled token
        """
        preds = np.asarray(seed).astype('float64')
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        probs = np.random.multinomial(1, exp_preds / np.sum(exp_preds), 1)
        return np.argmax(probs)

    def generate_script(self, seed, temp=0.5, **stop):
        """
        generate a script of certain length or until a token idx is reached.
        :param seed: np.array, input into model to generate sample from, dims (1, sample_len)
        :param temp: float, softmax temperature, amount of entropy to include in sample
        :param stop: kwargs, either len (int), number of new tokens to generate; or token (int) idx of token to stop at
        :return: list, full sequence generated as list of token indices (includes seed)
        """
        assert seed.shape[1] == self.sample_len, 'Incorrect sample length. Given: {}, Expecting: {}'.format(
            seed.shape[1], self.sample_len)
        generated_sequence = list(seed[0])

        while True:
            pred = self.model.predict(seed, verbose=0)[0]
            next_idx = self.sample_next_token(pred, temp)
            generated_sequence.append(next_idx)
            if stop.get('len', -1) == len(generated_sequence) - self.sample_len or stop.get('token', -1) == next_idx:
                break
            seed[0] = np_shift(seed[0], -1)
            seed[0, -1] = next_idx

        return generated_sequence


def np_shift(xs, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), xs[:-n]))
    else:
        return np.concatenate((xs[-n:], np.full(-n, np.nan)))

