import requests, re, pkgutil, numpy as np

class PythonPreprocessor:
  """
  Utility class to handling sourcing, cleaning, and encoding files.
  """
  def __init__(self, vocab_len=10000):
    self.vocab_len = vocab_len

  def merge_files(self):
    """
    merges all py files in data directory into 1 txt file.
    :return: int, number of bytes in txt file.
    """
    files = self.get_files()
    with open('data/merged.txt', 'w') as file:
      file.writelines(files)
    return sum([len(f) for f in files])

  def generate_training_data(self):
    corpus = self.get_files()
    max_len = len(max(corpus, key=len))
    x = np.zeros((len(corpus), max_len))
    y = np.zeros((len(corpus), ))

  def get_files(self):
    """
    gets all py files from data directory and returns their contents.
    :return: list, str's containing each files content
    """
    curr = self.read_file(num=0)
    files = []
    i = 0
    while curr:
      files.append(curr)
      i += 1
      curr = self.read_file(num=i, verbose=False)

    return files

  def source_local_file(self, source_path, num):
    """
    moves local file from given path to data directory with file_num label.
    :param source_path: str, path to source file
    :param num: int, number to label file with
    :return: None
    """
    self.write_file(self.read_file(path=source_path), num=num)
    self.clean_file(num=num)

  def source_raw_url(self, url, num):
    res = requests.get(url)
    if res.status_code == 200:
      self.write_file(res.content, num=num)
      self.clean_file(num=num)
    else:
      print(res)

  @staticmethod
  def read_file(**kwargs):
    assert 'num' in kwargs or 'path' in kwargs, 'Incorrect read_file usage, provide either path or num.'
    if 'num' in kwargs:
      kwargs['path'] = f'data/file{kwargs["num"]}.py'

    try:
      with open(kwargs['path'], 'r') as file:
        return file.read()
    except FileNotFoundError:
      if kwargs.get('verbose', True):
        print('INCORRECT FILE PATH:', kwargs['path'])
      return ''

  @staticmethod
  def write_file(data, **kwargs):
    assert 'num' in kwargs or 'path' in kwargs, 'Incorrect write_file usage, provide either path or num.'
    if 'num' in kwargs:
      kwargs['path'] = f'data/file{kwargs["num"]}.py'

    with open(kwargs['path'], 'w') as file:
      file.write(data)

  def clean_file(self, **kwargs):
    """
    Removes comments, remove bad imports, remove unnecessary newlines, add EOF token on given py file.
    provide either num (int) or path (str) as kwarg.
    :return: None
    """
    if 'num' in kwargs:
      kwargs['path'] = f'data/file{kwargs["num"]}.py'

    data = self.read_file(**kwargs)
    if not data:
      return

    # delete comments
    data = re.compile(r'""".*?"""', re.DOTALL).sub('', data)
    data = re.compile(r"'''.*?'''", re.DOTALL).sub('', data)
    data = re.compile('#.*').sub('', data)

    # delete bad imports and provide warnings
    import_re = re.compile(f'(\n|^)(import|from) (?!({"|".join(self.get_importable_modules())})).+')
    matches = import_re.finditer(data)
    has_bad_imports = False
    try:
      curr = next(matches)
      has_bad_imports = True
    except StopIteration:
      pass

    if has_bad_imports:
      print(f'WARNING: unavailable import in {kwargs.get("path", "data/file" + str(kwargs.get("num", 0)) + ".py")}.')
      while has_bad_imports:
        try:
          print('\t- "' + curr.group().strip('\n') + '" @ character index ' + str(curr.start()))
          curr = next(matches)
        except StopIteration:
          has_bad_imports = False
      data = import_re.sub('', data)

    # remove unnecessary newlines
    data = re.sub(r'\n[\n\W]+\n+', '\n', data)
    if data[0] == '\n':  # check for newline @ file start
      data = data[1:]

    # add special EOF token
    data = re.sub(' EOF ', '', data)  # deletes any old EOF tokens
    data += ' EOF '

    self.write_file(data, **kwargs)

  @staticmethod
  def get_importable_modules():
    """
    get a list of all importable modules in current venv.
    :return: list, list of strs, each of which is the name of an importable module
    """
    return [pkg.name for pkg in pkgutil.iter_modules()]

