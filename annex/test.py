from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
from datetime import datetime
import tensorflow as tf
import numpy as np

from process import image


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main(unused):

    p = image.Polygon([(0,0), (100, 0), (100, 100), (0, 100)])
    p.clip(point_min=(5, 5), point_max=(90, 90))
    print(p)


    channels = 3
    colorspace = b'RGB'

    image_features = tf.train.Features(feature={
        'image/height': _int64_feature(640),
        'image/width': _int64_feature(480),
        'image/channels': _int64_feature(channels),
        'image/colorspace': _bytes_feature(colorspace),

    })

    image_features.feature['masks'].bytes_list.value.append(b'crumpet')

    image_features.feature['captions'].bytes_list.value.extend([b'crow', b'cow'])

    objects_feature_list = [image_features]
    image_caption_list = []

    #dlorp = tf.train.FeatureList()
    florp = tf.train.FeatureList()

    fuckshit = tf.train.Features(feature={'test': _int64_feature(2)})

    dlorp = tf.train.FeatureList(feature=[])

    fl = florp.feature.add()
    fl.bytes_list.value.append(b'woop woop')
    fl = florp.feature.add()
    fl.bytes_list.value.append(b'thank you')

    image_feature_lists = tf.train.FeatureLists(
        feature_list={
            'objects': florp,
            'captions': dlorp
        }
    )

    example = tf.train.SequenceExample(
        context=image_features,
        feature_lists=image_feature_lists)

    example.feature_lists.feature_list['text'].feature.extend([_int64_feature([1,2,3,4])])
    example.feature_lists.feature_list['text'].feature.extend([_int64_feature([5,6,7,8])])

    print(example)


if __name__ == '__main__':
    tf.app.run()