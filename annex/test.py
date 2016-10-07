from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import skimage.io as io
import time
import timeit
import functools
import numbers

from process import processor_image
from external import coco
from record import Polygon2D, ClassLabel, BoundingBox
from record.example import *
from process import ProcessorImage
from dataset import dataset_factory


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'mscoco',
                           'Dataset type. One of ''mscoco'', ''imagenet''')

tf.app.flags.DEFINE_string('train_dir', '',
                           'Training data directory')

tf.app.flags.DEFINE_string('validation_dir', '',
                           'Validation data directory')

tf.app.flags.DEFINE_string('test_dir', '',
                           'Test data directory')


class Timer:
    def __init__(self, str=""):
        if str:
            print(str)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('Request took %.03f sec.' % self.interval)


def main(unused):

    bb = BoundingBox(1.0, 1.0, 300, 200)
    print(bb.is_integral())
    bbi = bb.as_integers()

    ds = dataset_factory.create(
        'mscoco',
        name='test',
        data_dir='/data/x/mscoco/train2014',
        annotation_file='/data/x/mscoco/annotations/instances_train2014.json')

    print("Num records: ", ds.num_records())
    # with Timer("Gen records"):
    #     all_recs2 = [x for x in dc.generate_records(include_objects=True)]
    #     print("Num generated: ", len(all_recs2))

    processor = ProcessorImage(ds, num_shards=256)
    processor.process_records()

if __name__ == '__main__':
    tf.app.run()