from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime
import threading
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

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


class Processor(object):
    def __init__(self, dataset, num_shards):
        self.dataset = dataset
        self._num_threads = FLAGS.num_threads
        self._num_shards = num_shards

    def _process_record(self, record, writer):
        print("Crap")

    def _process_records_batch(self, thread_index, thread_range, num_shards_batch, num_shards_total):
        """Processes and saves list of images as TFRecord in 1 thread.

        Args:
          thread_index: integer, unique batch to run index is within [0, len(ranges)).
          ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
          num_shards: integer number of shards for this data set.
        """

        shard_ranges = np.linspace(thread_range[0], thread_range[1], num_shards_batch + 1).astype(int)
        num_records_in_thread = thread_range[1] - thread_range[0]
        counter = 0
        print(num_shards_batch)
        for s in range(num_shards_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_batch + s
            output_filename = '%s-%.5d-of-%.5d' % (self.dataset.name, shard, num_shards_total)
            output_file = os.path.join(FLAGS.output_directory, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)
            shard_counter = 0
            for record in self.dataset.generate_records(start=shard_ranges[s], end=shard_ranges[s + 1]):
                self._process_record(record, writer)
                shard_counter += 1
                counter += 1
                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_records_in_thread))
                    sys.stdout.flush()

            print('%s [thread %d]: Wrote %d images to %s' %
                  (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %d shards.' %
              (datetime.now(), thread_index, counter, num_records_in_thread))
        sys.stdout.flush()

    def process_records(self):

        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(0, self.dataset.num_records(), self._num_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i+1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' % (self._num_threads, ranges))
        sys.stdout.flush()

        # Each thread produces N shards where N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2, then the first
        # thread would produce shards [0, 64).
        num_shards_per_batch = int(self._num_shards / len(ranges))

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()
        threads = []
        for thread_index in range(len(ranges)):
            args = (thread_index, ranges[thread_index], num_shards_per_batch, self._num_shards)
            t = threading.Thread(target=self._process_records_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d images in data set.' % (datetime.now(), self.dataset.num_records()))
        sys.stdout.flush()
