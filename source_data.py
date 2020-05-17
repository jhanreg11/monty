from scraper import *

if __name__ == '__main__':
  repos = [['keras', 'keras-team', 'keras'], ['scikit-learn', 'scikit-learn', 'sklearn'],
           ['Mask_RCNN', 'matterport', 'mrcnn'], ['face_recognition', 'ageitgey'],
           ['Detectron', 'facebookresearch', 'detectron'], ['pandas', 'pandas-dev', 'pandas'],
           ['matplotlib', 'matplotlib', 'src']]

  for args in repos:
    args.insert(2, 'data/buffer')
    try:
      s = Scraper(*args)
      s.run()
    except Exception as e:
      print('ERROR trying to run scraper', e)