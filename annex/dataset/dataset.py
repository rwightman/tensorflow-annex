from abc import abstractmethod
from os import path
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class Dataset(object):

    def __init__(self, name, data_dir=''):

        self.name = name

        assert path.isdir(data_dir)
        self.data_dir = data_dir

        self._records = {}
        self._records_index = []
