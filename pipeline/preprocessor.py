import numpy as np
import os
import re
import shutil
from pipeline.tokenizer import PyTokenizer
from utils import read_file, write_file

DEBUG = True


class PreProcessor:
    def __init__(self, buffer_dir, clean_file, tokenizer):
        """
        Create processor.
        :param buffer_dir: str, path to buffer directory
        :param clean_file: str, path to cleaned data file
        :param tokenizer: Tokenizer, tokenizer object
        """
        self.buffer_dir = buffer_dir
        self.clean_file = clean_file
        self.tokenizer = tokenizer

    def get_training_data(self, sample_len=50, step=1, one_hot_input=False):
        """
        get training data in form necessary for model training.
        :param sample_len: int, length of the samples to generate
        :param step: step to travel training sequence with
        :param one_hot_input: bool, whether to convert input to one hot vectors or not
        :return x: np.array, training inputs w/ dim (#samples, sample_len)
        :return y: np.array, training labels w/ dim (#samples, vocab_len)
        """
        if self.tokenizer.real_vocab_len == 0:
            data = read_file(self.clean_file)
            self.tokenizer.fit_on_data(data)

        data = read_file(self.clean_file, True)
        tokens = self.tokenizer.text_to_sequence(data)

        statements = []
        next_statements = []
        for i in range(0, len(tokens) - sample_len, step):
            statements.append(tokens[i:i + sample_len])
            next_statements.append(tokens[i + sample_len])

        if one_hot_input:
            raise NotImplementedError("Need to implement this")
        else:
            x = np.array(statements, dtype=np.int)

        y = np.zeros((len(statements), self.tokenizer.real_vocab_len), dtype=np.int)
        for i, next_statement in enumerate(next_statements):  # one hots y matrix
            y[i, next_statement] = 1

        if DEBUG:
            print('x shape:', x.shape, 'y shape:', y.shape)

        return x, y

    # DATA CLEANSING

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
                clean_data = self.process_text(data)
                write_file(clean_data, self.clean_file, True)
            except Exception:
                if DEBUG:
                    print('Error found tokenizing', path)

        if empty:
            shutil.rmtree(self.buffer_dir)
            os.mkdir(self.buffer_dir)

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


if __name__ == '__main__':
    t = PyTokenizer(3000)
    p = PreProcessor('data', 'data/clean.py', t)
    print(p.get_training_data(50, 1))
