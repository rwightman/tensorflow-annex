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

from external import coco
from record import Polygon2D, ClassLabel, BoundingBox
from record.example import *
from process import ProcessorImage
from dataset import dataset_factory


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