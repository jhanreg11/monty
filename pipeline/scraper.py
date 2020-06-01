import os
import shutil
from utils import copy_file, get_dir_length, read_file


class GithubScraper:
    """
    Scrape python files from github repo and save to buffer
    """

    def __init__(self, repo, owner, dest, search_root=''):
        self.repo = repo
        self.owner = owner
        self.dest = dest
        self.search_root = search_root

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
        os.system('git clone ' + repo_url + ' --depth 1')

    def search(self):
        file_paths = []
        for root, dirs, files in os.walk(os.path.join(self.repo, self.search_root)):
            file_paths += [os.path.join(root, f) for f in files if f.endswith('.py')]

        return file_paths

    @staticmethod
    def scrape_from_csv(filepath='../resources/repos.csv', dest='../data/buffer'):
        data = read_file(filepath)
        for line in data.split('\n'):
            if line == '':
                continue
            name, author = line.split(',')
            g = GithubScraper(name, author, dest)
            try:
                g.run()
            except Exception:
                print('Error scraping repo {} with name{}'.format(name, author))


if __name__ == '__main__':
    GithubScraper.scrape_from_csv()
