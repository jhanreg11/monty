import os, pkgutil, shutil

def read_file(path, verbose=False):
  """
  Read data from a file and return its contents in a string.
  :param path: str, path to file's location
  :param verbose: bool, whether to print error message
  :return: str, file's content or empty string if file not found.
  """
  try:
    with open(path, 'r') as file:
      return file.read()
  except FileNotFoundError:
    if verbose:
      print('INCORRECT FILE PATH:', path)
    return ''


def write_file(data, path, append=False):
  """
  Write information provided into file. overwrites all existing data and creates new file if necessary.
  :param data: str, information to write to file
  :param path: path to data's destination
  :param append: bool, whether to append or overwrite file
  :return: None
  """
  mode = 'w'
  if append:
    mode = 'a'

  with open(path, mode) as file:
    file.write(data)


def copy_file(source_path, dest_path):
  """
  Copies the content of a source file to either another arbitrary file path or to an index in the buffer.
  :param source_path: str, path to the source file
  :param dest_path: str, path to files destination
  :return: bool, success or failure
  """
  data = read_file(source_path)
  if data:
    write_file(data, dest_path)
    return True

  return False


def get_dir_length(path):
  """
  Gets number of files in buffer.
  :return: int, number of files in buffer directory
  """
  return len([0 for name in os.listdir(path) if os.path.isfile(name)])


def get_importable_modules():
  """
  get a list of all importable modules in current venv.
  :return: list, list of strs, each of which is the name of an importable module
  """
  modules = []
  for pkg in pkgutil.iter_modules():
    modules.append(pkg.name)

  return modules
