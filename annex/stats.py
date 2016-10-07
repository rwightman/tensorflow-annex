from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

from dataset import dataset_factory

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'mscoco',
                           """Dataset type. One of mscoco, imagenet""")


def main():

    ds = dataset_factory.create(
        FLAGS.dataset,
        data_dir='',
        annotation_file='')

    print("Stats for %s dataset at %s" % (ds.name, ds.data_dir))
    print("\tNum records:\t%s" % ds.num_records())

    #FIXME make stats generation part of dataset class hierarchy


if __name__ == '__main__':
    tf.app.run()