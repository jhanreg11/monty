from collections import OrderedDict
from io import BytesIO
from tokenize import tokenize, STRING, INDENT, DEDENT
from utils import *

DEBUG = False


class PyTokenizer:
    def __init__(self, max_vocab_len):
        """
        Create a tokenizer for python scripts.
        :param max_vocab_len: int, maximum size of vocabulary length. Actual length may be less
        """
        self.max_vocab_len = max_vocab_len
        self.word_idx = {}
        self.idx_word = {}

    def fit_on_data(self, data):
        """
        Create token index from data.
        :param data: str, corpus to create tokenizer on
        :return: None
        """
        wcounts = self.generate_wcounts(data)
        wcounts.insert(0, ['OOV', None])

        if len(wcounts) > self.max_vocab_len:
            wcounts = wcounts[:self.max_vocab_len]

        self.word_idx = dict(zip([wc[0] for wc in wcounts], list(range(len(wcounts)))))
        self.idx_word = dict(zip(list(range(len(wcounts))), [wc[0] for wc in wcounts]))

    def add_data_to_vocab(self, data):
        """
        Add new data to vocabulary without displacing old indices or generating duplicates
        :param data: str, data to tokenize and add
        :return: None
        """
        wcounts = self.generate_wcounts(data)
        curr_len = self.real_vocab_len
        for word, count in wcounts:
            if curr_len >= self.max_vocab_len:
                break
            elif word in self.word_idx:
                continue
            else:
                self.word_idx[word] = curr_len
                self.idx_word[curr_len] = word
                curr_len += 1

    @staticmethod
    def generate_wcounts(data):
        """
        utility method for creating word counts of each unique word in a list
        :param data: str, data to tokenize and work with
        :return: list, word counts, each element is form - [word (str), count (int)], sorted by count highest to lowest
        """
        tokens = PyTokenizer.py_tokenize(data)
        word_counts = OrderedDict()
        for t in tokens:
            if t in word_counts:
                word_counts[t] += 1
            else:
                word_counts[t] = 1

        wcounts = list(word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        return wcounts

    def text_to_sequence(self, text):
        """
        Convert string to sequence of token indices.
        :param text: str, text to tokenize
        :return: list, list of token indices
        """
        tokens = PyTokenizer.py_tokenize(text)
        return [self.word_idx.get(t, 1) for t in tokens]

    def sequence_to_text(self, seq):
        """
        Convert list of token indices to python string.
        :param seq: list, list of integer indices
        :return: str, joined token list
        """
        string_tokens = [self.idx_word.get(i, 'OOV') for i in seq]
        return PyTokenizer.py_untokenize(string_tokens)

    @property
    def real_vocab_len(self):
        """
        Get actual length of vocabulary
        :return: int, actual vocab length
        """
        return len(self.word_idx)

    @staticmethod
    def py_tokenize(data):
        """
        Convert py string into tokens.
        :param data: str, python script
        :return: list, list of string tokens.
        """
        token_generator = tokenize(BytesIO(data.encode('utf-8')).readline)
        tokens = []
        while True:
            try:
                token_type, val, start, end, line = next(token_generator)
            except Exception:
                break

            if DEBUG and '.' in val:
                print('In function py_tokenize. TOKEN_TYPE:', token_type, 'VALUE:', val, 'START_POS:', start,
                      'END_POS:', end, 'FULL_LINE:', line[:-1])

            if token_type == STRING:
                if val[0] != '"' and val[0] != "'":
                    str_contents = val[2:-1].split(' ')
                else:
                    str_contents = val[1:-1].split(' ')
                str_contents = [t for t in str_contents if t]
                tokens.extend(["'", *str_contents, "'"])
            elif token_type == INDENT:
                tokens.append('INDENT')
            elif token_type == DEDENT:
                tokens.append('DEDENT')
            elif val == 'utf-8':
                continue
            else:
                tokens.append(val)

        return tokens

    @staticmethod
    def py_untokenize(tokens):
        """
        Convert list of string tokens to single python script string
        :param tokens: list, list of strings
        :return: str, joined tokens
        """
        joined_tokens = ''
        indent = 0
        cont_str = False
        str_buffer = ''
        start_line = False
        num_lines = 1
        for i, t in enumerate(tokens):
            if start_line and t != 'INDENT' and t != 'DEDENT':
                joined_tokens += ' ' * indent
                start_line = False
            elif t == 'INDENT':
                indent += 4
                continue
            elif t == 'DEDENT':
                indent = max(0, indent - 4)
                continue

            if cont_str:
                if t == '"' or t == "'":
                    joined_tokens += str_buffer + t + ' '
                    str_buffer = ''
                    cont_str = False
                elif t == 'EOF':
                    if DEBUG:
                        print('ERROR: OPEN STRING WHEN EOF REACHED @ token', i, '@ line', num_lines,
                              'in Function: py_untokenize')
                    joined_tokens += str_buffer[0] * 2 + '\nEOF\n'
                    str_buffer = ''
                    cont_str = False
                else:
                    str_buffer += t + ' '
            elif t == "'" or t == '"':
                str_buffer += "'"
                cont_str = True
            elif t == '\n':
                start_line = True
                joined_tokens += '\n'
            elif t == 'EOF':
                num_lines += 1
                continue
            else:
                joined_tokens += t + ' '

        return joined_tokens

    def save(self, filepath):
        """
        Save tokenizer to a file.
        :param filepath: str,
        :return: None
        """
        with open(filepath, 'w') as file:
            for word in self.word_idx:
                file.write('{},{}\t'.format(word, self.word_idx[word]))

    @staticmethod
    def load(filepath):
        """
        Load tokenizer from file.
        :param filepath: str, filepath to load tokenizer from
        :return: PyTokenizer
        """
        tokenizer = PyTokenizer(100000)
        data = read_file(filepath)
        for line in data.split('\t'):
            if line == '':
                continue

            split = line.rindex(',')
            word = line[0:split]
            idx = int(line[split + 1:])

            tokenizer.word_idx[word] = idx
            tokenizer.idx_word[idx] = word

        return tokenizer


if __name__ == '__main__':
    t = PyTokenizer(5000)
    data = read_file('../data/clean.py')
    t.fit_on_data(data)

    # first_line = data[:300]
    # print('raw data:', first_line)
    # print('tokenized & untokenized data:', t.sequence_to_text(t.text_to_sequence(first_line)))

    t.save('../resources/tokenizer.csv')
    nt = PyTokenizer.load('../resources/tokenizer.csv')
    print(t.word_idx == nt.word_idx)

