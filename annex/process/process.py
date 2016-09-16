from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import threading
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


#metadata = (synsets, labels, humans, bboxes)

def _process_data_files_batch(
        processor, thread_index, ranges, name, num_shards,
        filenames, metadata):

    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      processor: instance of a concrete Processor.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
      bboxes: list of bounding boxes for each image. Note that each entry in this
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            metadata = metadata[i]

            record = processor.process_record(filename, metadata)

            example = record.to_example()

            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def process_data_files(
        name,
        processor,
        dataframe,
        num_shards):

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(dataframe.index), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    threads = []
    for thread_index in range(len(ranges)):
        args = (processor, thread_index, ranges, name, dataframe, num_shards)
        t = threading.Thread(target=_process_data_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(dataframe.index)))
    sys.stdout.flush()