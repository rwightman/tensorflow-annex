from abc import abstractmethod
from os import path
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'mscoco',
                           'Dataset type. One of ''mscoco'', ''imagenet''')

tf.app.flags.DEFINE_string('train_directory', '',
                           'Training data directory')

tf.app.flags.DEFINE_string('validation_directory', '',
                           'Validation data directory')

tf.app.flags.DEFINE_string('test_directory', '',
                           'Test data directory')

class Dataset(object):

    def __init__(self, name, data_dir=''):

        self.name = name

        assert path.isdir(data_dir)
        self.data_dir = data_dir

        self._records = {}

        self._initialize_metadata()

    @abstractmethod
    def _initialize_metadata(self):
        pass

    def process(self):
        self._processor.process()