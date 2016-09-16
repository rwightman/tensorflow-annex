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
                           """Dataset type. One of mscoco, imagenet""")


def main():

    if FLAGS.dataset == 'mscoco':
        dataset = DatasetCoco()
    elif FLAGS.dataset == 'imagenet':
        dataset = DatasetImagenet(synset_to_human, image_to_bboxes)
    else:
        dataset = DatasetImageRaw()



if __name__ == '__main__':
    tf.app.run()