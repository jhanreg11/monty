import os, shutil
from utils import copy_file, get_dir_length


class Scraper:
  """
  Scrape python files from github repo and save to buffer
  """
  def __init__(self, repo, owner, dest, root=''):
    self.repo = repo
    self.owner = owner
    self.dest = dest
    self.root = root

  def run(self):
    self.clone_repo()
    paths = self.search()
    idx = get_dir_length(self.dest)

    for path in paths:
      copy_file(path, os.path.join(self.dest, 'file{}.py'.format(idx)))
      idx += 1

    shutil.rmtree(self.repo)

  def clone_repo(self):
    repo_url = 'https://github.com/{}/{}.git'.format(self.owner, self.repo)
    os.system('git clone ' + repo_url)

  def search(self):
    file_paths = []
    for root, dirs, files in os.walk(os.path.join(self.repo, self.root)):
      file_paths += [os.path.join(root, f) for f in files if f.endswith('.py')]

    return file_paths
