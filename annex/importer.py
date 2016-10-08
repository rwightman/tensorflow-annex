# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

import os
import sys
from datetime import datetime
import tensorflow as tf
from dataset import dataset_factory
from process import ProcessorImage

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'mscoco',
                           'Dataset type. One of ''mscoco'', ''imagenet''')

tf.app.flags.DEFINE_string('train_dir', '',
                           'Training data directory')

tf.app.flags.DEFINE_string('validation_dir', '',
                           'Validation data directory')

tf.app.flags.DEFINE_string('test_dir', '',
                           'Test data directory')

tf.app.flags.DEFINE_string('output_dir', '',
                           'Output directory')

tf.app.flags.DEFINE_integer('train_shards', 512,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards', 512,
                            'Number of shards in test TFRecord files.')

tf.app.flags.DEFINE_string('annotation_file', '', 'Annotation file')


def main(argv):

    if not FLAGS.train_dir and not FLAGS.validation_dir and not FLAGS.test_dir:
        print("Nothing to do here")
        exit(1)

    dataset_type = FLAGS.dataset
    assert dataset_factory.is_valid_type(dataset_type)

    if FLAGS.train_dir:
        ds = dataset_factory.create(
            dataset_type,
            name='train',
            data_dir=FLAGS.train_dir,
            annotation_file=FLAGS.annotation_file)
        proc = ProcessorImage(ds, num_shards=FLAGS.train_shards, output_dir=FLAGS.output_dir)
        proc.process_records()

    if FLAGS.validation_dir:
        ds = dataset_factory.create(
            dataset_type,
            name='validation',
            data_dir=FLAGS.validation_dir,
            annotation_file=FLAGS.annotation_file)
        proc = ProcessorImage(ds, num_shards=FLAGS.validation_shards, output_dir=FLAGS.output_dir)
        proc.process_records()

    if FLAGS.test_directory:
        ds = dataset_factory.create(
            dataset_type,
            name='test',
            data_dir=FLAGS.test_dir,
            annotation_file=FLAGS.annotation_file)
        proc = ProcessorImage(ds, num_shards=FLAGS.test_shards, output_dir=FLAGS.output_dir)
        proc.process_records()


if __name__ == '__main__':
    tf.app.run()
