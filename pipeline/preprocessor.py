import numpy as np
import os
import re
import shutil
from collections import OrderedDict
from io import BytesIO
from tokenize import tokenize, TokenError, INDENT, DEDENT, STRING
from utils import read_file, write_file

DEBUG = True


class PreProcessor:

    def __init__(self, buffer_dir, clean_file, max_vocab_len):
        """
        Create processor.
        :param buffer_dir: str, path to buffer directory
        :param clean_file: str, path to cleaned data file
        :param max_vocab_len: int, maximum of size of vocabulary (actual size may be lower)
        """
        self.buffer_dir = buffer_dir
        self.clean_file = clean_file
        self.max_vocab_len = max_vocab_len
        self.word_idx = {}
        self.idx_word = {}

    def fit_on_data(self):
        """
        Create vocabulary from cleaned data file and assign to self.word_idx
        :return: None
        """
        data = read_file(self.clean_file)
        tokens = PreProcessor.py_tokenize(data)
        word_counts = OrderedDict()
        for t in tokens:
            if t in word_counts:
                word_counts[t] += 1
            else:
                word_counts[t] = 1

        wcounts = list(word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        wcounts.insert(0, ['OOV', None])

        if len(wcounts) > self.max_vocab_len:
            wcounts = wcounts[:self.max_vocab_len]

        self.word_idx = dict(zip([wc[0] for wc in wcounts], list(range(len(wcounts)))))
        self.idx_word = dict(zip(list(range(1, len(wcounts) + 1)), [wc[0] for wc in wcounts]))

    def clean_buffer(self, empty=True, append=True):
        """
        clean all files in buffer and add to cleaned data file and empty buffer if necessary.
        :param empty: bool, whether to empty buffer or not.
        :param append: bool, whether to append cleaned data into clean file or overwrite it.
        :return: None
        """
        write_file('', self.clean_file, append)  # clear old file if necessary
        for i, file in enumerate(os.listdir(self.buffer_dir)):
            path = os.path.join(self.buffer_dir, file)
            data = read_file(path)
            if not data:
                os.remove(path)
                continue
            try:
                self.py_tokenize(data)
                clean_data = self.process_text(data)
                write_file(clean_data, self.clean_file, True)
            except TokenError:
                if DEBUG:
                    print('Error found tokenizing', path)

        if empty:
            shutil.rmtree(self.buffer_dir)
            os.mkdir(self.buffer_dir)

    def text_to_sequence(self, text):
        """
        Convert string to sequence of token indices.
        :param text: str, text to tokenize
        :return: list, list of token indices
        """
        tokens = PreProcessor.py_tokenize(text)
        return [self.word_idx.get(t, 1) for t in tokens]

    def sequence_to_text(self, seq):
        """
        Convert list of token indices to python string.
        :param seq: list, list of integer indices
        :return: str, joined token list
        """
        string_tokens = [self.idx_word.get(i, 'OOV') for i in seq]
        return PreProcessor.py_untokenize(string_tokens)

    def get_training_data(self, sample_len=50, step=1):
        """
        get training data in form necessary for model training.
        :param sample_len: int, length of the samples to generate
        :param step: step to travel training sequence with
        :return x: np.array, training inputs w/ dim (#samples, sample_len)
        :return y: np.array, training labels w/ dim (#samples, vocab_len)
        """
        if len(self.word_idx) == 0:
            self.fit_on_data()
        print(self.word_idx)

        data = read_file(self.clean_file, True)
        tokens = self.text_to_sequence(data)

        statements = []
        next_statements = []
        for i in range(0, len(tokens) - sample_len, step):
            statements.append(tokens[i:i + sample_len])
            next_statements.append(tokens[i + sample_len])

        x = np.array(statements, dtype=np.int)
        y = np.zeros((len(statements), len(self.word_idx)), dtype=np.int)
        if DEBUG:
            print('x shape:', x.shape, 'y shape:', y.shape)
        for i, next_statement in enumerate(next_statements):  # one hots y matrix
            y[i, next_statement] = 1

        return x, y

    @staticmethod
    def py_tokenize(data, cleaning=False):
        token_generator = tokenize(BytesIO(data.encode('utf-8')).readline)

        if cleaning:
            return token_generator

        tokens = []
        print_next = False
        i = 0
        while True:
            # if token_type == INDENT:
            #   continue
            # if prev_type == NEWLINE and start[1]:  # if this is the first token in a line and it doesn't start @ col 0
            #   tokens.extend(['    ' for _ in range(start[1] // PreProcessor.indent_size)])
            try:
                token_type, val, start, end, line = next(token_generator)
            except Exception:
                break

            # if DEBUG and ("No data provided for" in val or (print_next and i < 40)):
            #     print('In function py_tokenize. TOKEN_TYPE:', token_type, 'VALUE:', val, 'START_POS:', start,
            #           'END_POS:', end, 'FULL_LINE:', line[:-1])
            #     print_next = True
            #     i += 1

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


    @property
    def real_vocab_len(self):
        return len(self.word_idx)

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
                    last_10 = joined_tokens[-10:]
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

    @staticmethod
    def process_text(data):
        """
        Clean a text string of python code.
        :param data: str, python code
        :return: str, cleaned python code
        """
        # delete comments
        def comment_subber(match_obj):
            string = match_obj.group(0)
            if string.startswith("'''") or string.startswith('"""') or string.startswith('#'):
                return ''
            return string

        comment_pattern = '""".*?"""|\'\'\'.*?\'\'\'|"(\\[\s\S]|[^"])*"|\'(\\[\s\S]|[^\'])*\'|#[\s\S]*'
        data = re.compile(comment_pattern, re.DOTALL).sub(comment_subber, data)

        # remove imports
        data = re.sub('(\n|^)(import|from).*', '', data)

        # add special EOF token
        data = re.sub('[\n\s]*EOF[\n\s]*', '', data)  # deletes any old EOF tokens
        data += '\nEOF\n'

        # remove unnecessary newlines
        data = PreProcessor.remove_newlines(data)

        return data

    @staticmethod
    def remove_newlines(data):
        """
    Remove unnecessary newlines from python string
    :param data: str, python code
    :return: str, cleaned python code
    """
        data = re.sub(r'\n[\n\s]*\n', '\n', data)
        while data[0] == '\n':  # check for newline @ file start
            data = data[1:]

        return data


test_script = """def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
  if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
      hasher = hashlib.sha256()
  else:
      hasher = hashlib.md5()
  with open(fpath, 'rb') as fpath_file:
      for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
          hasher.update(chunk)
  return hasher.hexdigest() EOF"""

if __name__ == '__main__':
    # test_file = 'data/test.py'
    # out_file = 'data/test_out.py'
    #
    # p = PreProcessor('data/buffer', 'data/clean.py', 10000)
    #
    # p.clean_buffer(empty=False, append=False, verbose=True)
    # p.fit_on_data()
    # seq = p.text_to_sequence(read_file(p.clean_file))
    # text = p.sequence_to_text(seq)
    # write_file(text, out_file)

    # tokens = p.py_tokenize(p.process_text(read_file(test_file)))
    # result = p.py_untokenize(tokens)
    # write_file(result, out_file)

    p = PreProcessor('data', 'data/clean.py', 10000)
    print(p.get_training_data(50, 1))
