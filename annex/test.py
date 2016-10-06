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

from process import process_image
from external import coco
from record import Polygon2D, ClassLabel, BoundingBox
from record.example import *
import dataset
from dataset import DatasetCoco

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

    dc = DatasetCoco(
        data_dir='/data/x/mscoco/train2014',
        annotation_file='/data/x/mscoco/annotations/instances_train2014.json')

    print("Num records: ", dc.num_records())
    with Timer("Gen records"):
        all_recs2 = [x for x in dc.generate_records(include_objects=True)]
        print("Num generated: ", len(all_recs2))

    serialized = []
    with Timer("Convert records"):
        for x in dc.generate_records(include_objects=True):
            ex = x.to_example()
            serialized.append(ex)



if __name__ == '__main__':
    tf.app.run()