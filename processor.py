import os, re, shutil
from utils import get_importable_modules, read_file, write_file

class Processor:
  def __init__(self, buffer_path, data_path):
    """
    Create processor.
    :param buffer_path: str, path to buffer directory.
    :param data_path: str, path to txt file to store cleaned data
    """
    self.buffer_path = buffer_path
    self.clean_path = data_path

  def clean_buffer(self):
    """
    clean all files in buffer and add to cleaned data file and empty buffer.
    :return: None
    """
    clean_data = ''
    for file in os.listdir(self.buffer_path):
      path = os.path.join(self.buffer_path, file)
      clean_data += self.clean_data(read_file(path))

    write_file(clean_data, self.clean_path, append=True)

    shutil.rmtree(self.buffer_path)
    os.mkdir(self.buffer_path)

  @staticmethod
  def py_tokenize(data):
    operators = [':', '(', ')', '{', '}', '[', ']', "'", '"', ',', '+', '-', '*', '/', '%', '=', '>', '<', '&', '|',
                 '~', '^', '**', '//', '>>', '<<', '+=', '-=', '*=', '%=', '&=', '|=', '^=', '==', '!=', '/=', '**=',
                 '>>=', '<<=']


  @staticmethod
  def simple_tokenize(text, filters='\/?#`', delim=' \n'):
    return re.split(r'([\W]|\n)', ''.join(filter(lambda c: c not in filters, text)))

  @staticmethod
  def clean_data(data):
    # delete comments
    data = re.compile(r'""".*?"""', re.DOTALL).sub('', data)
    data = re.compile(r"'''.*?'''", re.DOTALL).sub('', data)
    data = re.compile('#.*').sub('', data)

    # delete bad imports and provide warnings
    data = re.sub(f'(\n|^)(import|from) (?!({"|".join(get_importable_modules())})).+', '', data)

    # remove unnecessary newlines
    data = re.sub(r'\n[\n\W]+\n+', '\n', data)
    if data[0] == '\n':  # check for newline @ file start
      data = data[1:]

    # add special EOF token
    data = re.sub('[\n\W]*EOF[\n\W]*', '', data)  # deletes any old EOF tokens
    data += '\nEOF\n'

    return data