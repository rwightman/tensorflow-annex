from abc import abstractmethod
from os import path
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Dataset(object):

    def __init__(self, name, data_dir=''):

        self.name = name

        assert path.isdir(data_dir)
        self._data_dir = data_dir

        self._records = {}

        self._initialize_metadata()

    @abstractmethod
    def _initialize_metadata(self):
        pass

    def process(self):
        self._processor.process()