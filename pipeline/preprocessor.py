import os
import re
import shutil
from pipeline.tokenizer import PyTokenizer
from utils import read_file, write_file

DEBUG = True


class PreProcessor:
    def __init__(self, buffer_dir, clean_file):
        """
        Create processor.
        :param buffer_dir: str, path to buffer directory
        :param clean_file: str, path to cleaned data file
        """
        self.buffer_dir = buffer_dir
        self.clean_file = clean_file

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
                write_file(clean_data, self.clean_file, append=True)
            except Exception:
                if DEBUG:
                    print('Error found tokenizing', path)

        if empty:
            shutil.rmtree(self.buffer_dir)
            os.mkdir(self.buffer_dir)

    @staticmethod
    # TODO: replace contents of strings with random words from a small word bank
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

        comment_pattern = '\"\"\"(.|\n)*?\"\"\"|\'\'\'(.|\n)*?\'\'\'|#.[^\n]*'
        data = re.compile(comment_pattern, re.DOTALL).sub(comment_subber, data)

        # remove imports
        data = re.sub('(\n|^)(import|from).*', '', data)

        # remove decorators
        data = re.sub('@.*\n', '', data)

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
    p = PreProcessor('data/buffer', 'data/clean.py', t)
    p.clean_buffer(False, False)
