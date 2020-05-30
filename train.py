import argparse
from pipeline.model import Model
from pipeline.preprocessor import PreProcessor
from pipeline.tokenizer import PyTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--max_vocab_len', type=int, default=2000)
    parser.add_argument('--sample_len', type=int, default=50, help='length of each training sample and sampling seed.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--buffer_dir', type=str, default='data/buffer')
    parser.add_argument('--clean_path', type=str, default='data/clean.py', help='path to file containing cleansed data')
    args = parser.parse_args()

    t = PyTokenizer(args.max_vocab_len)
    p = PreProcessor(args.buffer_dir, args.clean_path, t)
    x, y = p.get_training_data(args.sample_len)

    m = Model(t.real_vocab_len, args.sample_len)
    m.train(x, y, args.epochs, args.batch_size)