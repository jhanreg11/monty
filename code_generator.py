from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
import requests, numpy as np, re, pkgutil

###############
# GLOBAL VARS #
###############

vocab_len = 10000


###########################
# DATA SOURCING FUNCTIONS #
###########################

def create_file_local(source_loc, file_num):
  """
  moves local file from given path to data directory with file_num label.
  :param source_loc: str, path to source file
  :param file_num: int, number to label file with
  :return: None
  """
  with open(source_loc, 'r') as infile:
    lines = infile.readlines()

  with open('data/file' + str(file_num) + '.py', 'w') as outfile:
    outfile.writelines(lines)

  return file_num + 1


def dir_load_61a(dir, start_idx, file_num, clip_end=0, dir_start_idx=1):
  basepath = '/Users/jacobhanson/Desktop/CS-61A'
  fails = 0
  for i in range(dir_start_idx, file_num + dir_start_idx):
    try:
      num = '0' + str(i) if i < 10 else str(i)
      create_file_local(f'{basepath}/{dir}/{dir[:len(dir) - clip_end]}{num}/{dir[:len(dir) - clip_end]}{num}.py',
                start_idx + i - fails)
    except FileNotFoundError as e:
      fails += 1
      print('File not found @ idx', i, 'total fails:', fails)

  return start_idx + file_num - fails


def create_file_url(url, file_num):
  """
  Load raw file from url into data with file_num label.
  :param url: url to raw file
  :param file_num: number to label file w/ locally
  :return: None
  """
  res = requests.get(url)
  if res.status_code == 200:
    with open('data/file' + str(file_num) + '.py', 'w') as file:
      file.write(res.content.decode('utf-8'))
  else:
    print(res)


def load_file(num):
  """
  returns file contents in string.
  :param num: int, number of the file to load
  :return: str, file content
  """
  try:
    with open(f'data/file{num}.py', 'r') as file:
      return file.read()
  except FileNotFoundError:
    print('ERROR: incorrect file num given to load_file:', num)
    return ''

def write_file(num, data):
  with open(f'data/file{num}.py', 'w') as file:
    file.write(data)

#################################
# FILE PRE-PROCESSING FUNCTIONS #
#################################

def simple_tokenize(text, filters='\/?#`', delim=' \n'):
  return re.split(r'([\W]|\n)', ''.join(filter(lambda c: c not in filters, text)))

def py_split_token(tok, operators):
  def helper(token):
    if not token:
      return []
    for op in operators:
      if op in token:
        start_idx = token.index(op)
        end_idx = start_idx + len(op)
        return helper(token[:start_idx].strip()) + [op] + helper(token[end_idx:].strip())
    return [token]

  return helper(tok)

def py_to_tokens(text, filters='', lower=True, split=' '):
  """
  converts python text to list of tokens. Used to replace keras' text_to_word_sequences,
  :param text: str, python text
  :return: list, list of str tokens
  """
  operators = [c for c in ':(){}[]\'",+-*/%=><&|~^'] + ['**', '//', '>>', '<<'] + \
              [char + '=' for char in [c for c in '+-*%&|^=!'] + ['/', '**', '>>', '<<']]
  operators.sort(key=len)

  natl_tokens = simple_tokenize(text)
  tokens = []
  for token in natl_tokens:
    tokens += py_split_token(token, operators)

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


def encode_files():
  """
  creates word index for files in data directory, and encodes each file using word index.
  :return: list, list of all files with sequence lists
  """
  done = False
  file_num = 0
  file_strings = []

  while not done:
    try:
      with open(f'data/file{file_num}.py', 'r') as file:
        raw = file.readlines()
      file_strings.append(''.join(raw))
      file_num += 1
    except FileNotFoundError:
      done = True

  tokenizer = Tokenizer(num_words=vocab_len, filters=r'\/?;', lower=False, oov_token=1)
  tokenizer.fit_on_texts(file_strings)
  return tokenizer.texts_to_sequences(file_strings), tokenizer


def clean_file(num):
  """
  Removes comments, remove bad imports, remove unnecessary newlines, add EOF token on py file in data directory.
  :param num: file number to clean
  :return: None
  """
  # read file
  file_path = 'data/file' + str(num) + '.py'
  try:
    with open(file_path, 'r') as file:
      data = file.read()
  except FileNotFoundError as e:
    print("ERROR: Incorrect file num given")
    print(e)
    return

  # delete comments
  data = re.compile(r'""".*?"""', re.DOTALL).sub('', data)
  data = re.compile(r"'''.*?'''", re.DOTALL).sub('', data)
  data = re.compile('#.*').sub('', data)

  # delete bad imports and provide warnings
  import_re = re.compile(f'(\n|^)(import|from) (?!({"|".join(get_importable_modules())})).+')
  matches = import_re.finditer(data)
  has_bad_imports = False
  try:
    curr = next(matches)
    has_bad_imports = True
  except StopIteration:
    pass

  if has_bad_imports:
    print(f'WARNING: unavailable import in file{num}.py. Bad Statements:')
    while has_bad_imports:
      try:
        print('\t- "' + curr.group().strip('\n') + '" @ character index ' + str(curr.start()))
        curr = next(matches)
      except StopIteration:
        has_bad_imports = False
    data = import_re.sub('', data)

  # remove unnecessary newlines
  data = re.sub(r'\n[\n\W]+\n+', '\n', data)
  if data[0] == '\n': # check for newline @ file start
    data = data[1:]

  # add special EOF token
  data = re.sub(' EOF', '', data)  # deletes any old EOF tokens
  data += ' EOF'

  print(data)
  # write clean data to file
  with open(file_path, 'w') as file:
    file.write(data)


def get_importable_modules():
  """
  get a list of all importable modules in current venv.
  :return: list, list of strs, each of which is the name of an importable module
  """
  modules = []
  for pkg in pkgutil.iter_modules():
    modules.append(pkg.name)

  return modules


###################
# MODEL FUNCTIONS #
###################

def create_model(**kwargs):
  if 'embed_dim' not in kwargs:
    kwargs['embed_dim'] = 32
  if 'lr' not in kwargs:
    kwargs['lr'] = .01

  model = Sequential()
  model.add(Embedding(vocab_len, kwargs['embed_dim']))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dense(1000, activation='sigmoid'))
  model.add(Dense(vocab_len, activation='softmax'))
  optimizer = RMSprop(lr=kwargs['lr'])
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

  return model


def create_model_fn(**kwargs):
  if 'embed_dim' not in kwargs:
    kwargs['embed_dim'] = 32
  if 'lr' not in kwargs:
    kwargs['lr'] = .01

  model = Input()


def sample(model):
  # TODO: sample model to generate python file
  pass


if __name__ == '__main__':
  text_to_word_sequence = py_to_tokens
  print(py_to_tokens(load_file(0)))
