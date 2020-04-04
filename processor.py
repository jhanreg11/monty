import os, re, shutil, numpy as np
from tokenize import tokenize, untokenize
from io import BytesIO
from utils import get_importable_modules, read_file, write_file, get_dir_length
from collections import OrderedDict


class Processor:
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

  def create_word_idx(self):
    """
    Create vocabulary from cleaned data file and assign to self.word_idx
    :return: None
    """
    data = read_file(self.clean_file)
    tokens = Processor.py_tokenize(data)
    word_counts = OrderedDict()
    for t in tokens:
      if t in word_counts:
        word_counts[t] += 1
      else:
        word_counts[t] = 1

    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    wcounts.prepend(['OOV', None])

    if len(wcounts) > self.max_vocab_len:
      wcounts = wcounts[:self.max_vocab_len]

    self.word_idx = dict(zip([wc[0] for wc in wcounts], list(range(1, len(wcounts) + 1))))

  def empty_buffer(self):
    """
    clean all files in buffer and add to cleaned data file and empty buffer.
    :return: None
    """
    for i, file in enumerate(os.listdir(self.buffer_dir)):
      path = os.path.join(self.buffer_dir, file)
      clean_data = self.process_text(read_file(path))
      write_file(clean_data, self.clean_file, 'a')

    shutil.rmtree(self.buffer_dir)
    os.mkdir(self.buffer_dir)

  def text_to_sequence(self, text):
    """
    Convert string to sequence of token indices.
    :param text: str, text to tokenize
    :return: list, list of token indices
    """
    tokens = Processor.py_tokenize(text)
    return [self.word_idx.get(t, 1) for t in tokens]

  def get_training_data(self, sample_len=50, step=1):
    """
    get training data in form necessary for model training.
    :param sample_len: int, length of the samples to generate
    :param step: step to travel training sequence with
    :return x: np.array, training inputs w/ dim (#samples, sample_len)
    :return y: np.array, training labels w/ dim (#samples, vocab_len)
    """
    data = read_file(self.clean_file)
    tokens = self.text_to_sequence(data)

    statements = []
    next_statements = []
    for i in range(0, len(tokens) - sample_len, step):
      statements.append(tokens[i:i+sample_len])
      next_statements.append(tokens[i+sample_len])

    x = np.array(statements, dtype=np.int)
    y = np.zeros((len(statements), len(self.word_idx)), dtype=np.int)
    for i, next_statement in enumerate(next_statements):  # one hots y matrix
      y[i, next_statement] = 1

    return x, y

  @staticmethod
  def py_tokenize(data, full=False):
    if full:
      return tokenize(BytesIO(data.encode('utf-8')).readline)
    return [str_tok for _, str_tok, _, _, _ in tokenize(BytesIO(data.encode('utf-8')).readline)][1:]

  @staticmethod
  def py_tokenize_old(data):
    """
    convert python code string to list of string tokens
    :param data: str, python code
    :return: list, list of string tokens
    """
    operators = ['<<=', '>>=', '**=', '/=', '!=', '==', '^=', '|=', '&=', '%=', '*=', '-=', '+=', '<<', '>>', '//',
                 '**', '^', '~', '|', '&', '<', '>', '=', '%', '/', '*', '-', '+', ',', '"', "'", ']', '[', '}', '{',
                 ')', '(', ':']

    tokens = Processor.simple_tokenize(data)
    i = 0
    while i < len(tokens):
      if tokens[i] == '':
          tokens.pop(i)
          continue

      for op in operators:
        if op == tokens[i]:
          break

        if op in tokens[i]:
          a = i  # temporary idx
          old = tokens.pop(a)  # remove old multi-token
          before_op = old[:old.index(op)]  # get everything before operator
          if before_op:
            tokens.insert(a, before_op)
            a += 1
          tokens.insert(a, op)
          after_op = old[old.index(op) + len(op):]  # get everything after operator
          if after_op:
            tokens.insert(a+1, after_op)

          i -= 1  # done so i reads before_op/op in next iteration
          break

      i += 1

    consec_spaces = 0
    i = 0
    while i < len(tokens):
      if tokens[i] == ' ':
        consec_spaces += 1
      else:
        consec_spaces = 0

      if consec_spaces == 4:
        tokens[i] = '\t'
        [tokens.pop(i - j) for j in range(1, 4)]
        i -= 3
        consec_spaces = 0

      i += 1

    return tokens

  @staticmethod
  def simple_tokenize(text, filters='\/?#`'):
    """
    simple function to convert string to string token list
    :param text: str, text to tokenize
    :param filters: str, characters to filter from text
    :return: list, list of string tokens
    """
    return re.split(r'([\W]|\n)', ''.join(filter(lambda c: c not in filters, text)))

  @staticmethod
  def process_text(data):
    tokens = Processor.py_tokenize(data, full=True)
    return untokenize([t for t in tokens if t and '#' != t[0] and '"""' != t[:3]]).decode('utf-8')

  @staticmethod
  def process_text_old(data):
    """
    Clean a text string of python code.
    :param data: str, python code
    :return: str, cleaned python code
    """
    # delete comments
    data = re.compile(r'""".*?"""', re.DOTALL).sub('', data)
    data = re.compile(r"'''.*?'''", re.DOTALL).sub('', data)
    data = re.compile('#.*').sub('', data)

    # delete bad imports and provide warnings
    data = re.sub(f'(\n|^)(import|from) (?!({"|".join(get_importable_modules())})).+', '', data)

    # delete decorators
    data = re.sub(r'\n\s*@.*\n', '\n', data)

    # remove unnecessary newlines
    data = Processor.remove_newlines(data)

    # add special EOF token
    data = re.sub('[\n\W]*EOF[\n\W]*', '', data)  # deletes any old EOF tokens
    data += '\nEOF\n'

    return data

  @staticmethod
  def remove_newlines(data):
    """
    Remove unnecessary newlines from python string
    :param data: str, python code
    :return: str, cleaned python code
    """
    data = re.sub(r'\n[\n\W]+\n+', '\n', data)
    while data[0] == '\n':  # check for newline @ file start
      data = data[1:]

    return data





