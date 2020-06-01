import argparse
from pipeline.model import Model
from pipeline.data_loader import DataLoader
from pipeline.tokenizer import PyTokenizer
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--max_vocab_len', type=int, default=2000)
    parser.add_argument('--sample_len', type=int, default=50, help='length of each training sample and sampling seed.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--clean_path', type=str, default='data/clean.py', help='path to file containing cleansed data')
    parser.add_argument('--pretrained_path', type=str, default='')
    args = parser.parse_args()

    t = PyTokenizer(args.max_vocab_len)
    dl = DataLoader(t, sample_len=args.sample_len, data_file=args.clean_path)
    x, y = dl.get_training_data()

    m = Model(t.real_vocab_len, args.sample_len)
    if args.pretrained_path:
        m.model = load_model(args.pretrained_path)

    m.train(x, y, dl.get_test_data(), args.epochs, args.batch_size)
