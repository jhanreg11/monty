import argparse
import numpy as np
from tensorflow.keras.models import load_model
from pipeline.model import Model
from pipeline.tokenizer import PyTokenizer
from utils import read_file


def generate_seed(tokenizer, sample_len):
    """
    Get seed from cleaned file.
    :param tokenizer: PyTokenizer, tokenizer to use.
    :param sample_len: int, length of seed to generate
    :return: np.array, model seed shape (1, sample_len)
    """
    tokens = []
    with open('data/test.py', 'r') as file:
        while len(tokens) < sample_len:
            tokens.extend(tokenizer.text_to_sequence(file.readline()))

    if len(tokens) != sample_len:
        tokens = tokens[:sample_len]

    return np.asarray(tokens, dtype=np.int).reshape((1, sample_len))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample scripts from saved model')
    parser.add_argument('--sample_len', type=int, default=-1, help='-1 if sample till end of file. otherwise it is the '
                                                                   'number of tokens past the initial seed to sample.')
    parser.add_argument('--model_path', type=str, default='best_model', help='path to model')
    args = parser.parse_args()

    t = PyTokenizer(5000)
    data = read_file('data/clean.py')
    t.fit_on_data(data)
    m = Model(t.real_vocab_len, 50)
    m.model = load_model(args.model_path)
    seed = generate_seed(t, 50)
    sample = m.generate_script(seed, token=t.word_idx['EOF'])
    print(t.sequence_to_text(sample))
