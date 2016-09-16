from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
from datetime import datetime
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('dataset', 'mscoco',
                           'Dataset type. One of ''mscoco'', ''imagenet''')

tf.app.flags.DEFINE_string('train_directory', '',
                           'Training data directory')

tf.app.flags.DEFINE_string('validation_directory', '',
                           'Validation data directory')

tf.app.flags.DEFINE_string('test_directory', '',
                           'Test data directory')

tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards', 1024,
                            'Number of shards in test TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')


def main():

    if not FLAGS.train_directory \
            and not FLAGS.validation_directory \
            and not FLAGS.test_directory:
        print("Nothing to do here")
        exit(0)

    if FLAGS.dataset == 'mscoco':
        dataset = DatasetCoco()
    elif FLAGS.dataset == 'imagenet':
        dataset = DatasetImagenet(synset_to_human_path, bbox_path)
    else:
        dataset = DatasetImageRaw()


    if FLAGS.train_directory:
        dataset.process(
            'train', FLAGS.train_directory, FLAGS.train_shards)

    if FLAGS.validation_directory:
        dataset.process(
            'validation', FLAGS.validation_directory, FLAGS.validation_shards)

    if FLAGS.test_directory:
       dataset.process(
            'test', FLAGS.test_directory, FLAGS.validation_shards)



if __name__ == '__main__':
    tf.app.run()
