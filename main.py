from model import Model
from processor import Processor
from utils import read_file, write_file

if __name__ == '__main__':
  write_file(Processor.process_text(read_file('data/cleaned_data.txt')), 'data/cleaned_data.txt')
  p = Processor('data/buffer', 'data/cleaned_data.txt', 100)
  p.create_word_idx()
  print(p.word_idx)
  # print(Processor.py_tokenize(Processor.process_text(read_file('data/cleaned_data.txt')[:1000])))

  # cleaned_text = Processor.process_text(read_file('data/test.txt'))
  # tokens = Processor.py_tokenize(cleaned_text)
  # print(tokens)