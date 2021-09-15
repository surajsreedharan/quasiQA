
import logging


from preprocess import preprocess


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  preprocess.run()