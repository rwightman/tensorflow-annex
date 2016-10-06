import numbers
import tensorflow as tf
import numpy as np
from enum import Enum


class FeatureKind(Enum):
    BYTES_LIST = 1,  # explicit option for list of bytes due to bytes/buffers having iterator interface list list
    BYTES = 2,  # one bytes value
    FLOAT = 3,  # covers single float and list of floats
    INT64 = 4   # covers single integer and list of integers


def _convert_bytes(value):
    assert not isinstance(value, numbers.Integral)  # integer is a valid bytes init value, so force error here
    if isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        value = bytes(value, encoding='utf8')
    else:
        value = bytes(value)
    return value


def _convert_bytes_list(value_list):
    if not value_list:
        return []
    elif isinstance(value_list, list) and isinstance(value_list[0], (bytes, bytearray)):
        return value_list  # skip iteration, assume all elements are also bytes
    return [_convert_bytes(v) for v in value_list]


def feature_int64(value_list):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def feature_float(value_list):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def feature_bytes(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = _convert_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_bytes_list(value_list, skip_convert=False):
    """Wrapper for inserting bytes features into Example proto."""
    if not skip_convert:
        value_list = _convert_bytes_list(value_list)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def feature_select(value, kind: FeatureKind):
    if kind == FeatureKind.BYTES_LIST:
        return feature_bytes_list(value)
    elif kind == FeatureKind.BYTES:
        return feature_bytes(value)
    elif kind == FeatureKind.FLOAT:
        return feature_float(value)
    else:
        return feature_int64(value)


def add_feature_int64(features, key=None, value=None):
    if not isinstance(value, list):
        value = [value]

    if isinstance(features, tf.train.Features):
        assert key
        features.feature[key].int64_list.value.append(value)
    elif isinstance(features, tf.train.FeatureList):
        assert not key
        features.feature.add().int64_list.value.append(value)
    elif isinstance(features, tf.train.FeatureLists):
        assert key
        features.feature_list[key].feature.extend([feature_int64(value)])
    else:
        assert False, "Invalid features type %s" % type(features)
    return features


def add_feature_float(features, key=None, value=None):
    if not isinstance(value, list):
        value = [value]

    if isinstance(features, tf.train.Features):
        assert key
        features.feature[key].float_list.value.append(value)
    elif isinstance(features, tf.train.FeatureList):
        assert not key
        features.feature.add().float_list.value.append(value)
    elif isinstance(features, tf.train.FeatureLists):
        assert key
        features.feature_list[key].feature.extend([feature_int64(value)])
    else:
        assert False, "Invalid features type %s" % type(features)
    return features


def add_feature_bytes(features, key=None, value=None):
    value = _convert_bytes(value)
    if isinstance(features, tf.train.Features):
        assert key
        features.feature[key].bytes_list.value.append(value)
    elif isinstance(features, tf.train.FeatureList):
        assert not key
        features.feature.add().bytes_list.value.append(value)
    elif isinstance(features, tf.train.FeatureLists):
        assert key
        features.feature_list[key].feature.extend([feature_bytes(value)])
    else:
        assert False, "Invalid features type %s" % type(features)
    return features


def add_feature_bytes_list(features, key=None, value_list=None):
    value_list = _convert_bytes_list(value_list)
    if isinstance(features, tf.train.Features):
        features.feature[key].bytes_list.value.extend(value_list)
    elif isinstance(features, tf.train.FeatureList):
        features.feature.add().bytes_list.value.extend(value_list)
    elif isinstance(features, tf.train.FeatureLists):
        assert key
        features.feature_list[key].feature.extend([feature_bytes_list(value_list)])
    else:
        assert False, "Invalid features type %s" % type(features)
    return features


class FeatureInt64(object):
    def __init__(self, value=[]):
        if not isinstance(value, list):
            value = [value]
        self.value = value

    def as_feature(self):
        return feature_int64(self.value)

    def add_to(self, dest, key=None):
        return add_feature_int64(dest, key, self.value)


class FeatureFloat(object):
    def __init__(self):
        self.value = []


class FeatureBytes(object):
    def __init__(self):
        self.value = bytes()


class FeatureBytesList(object):
    def __init__(self):
        self.value = []