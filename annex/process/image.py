from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


###


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


class ObjClass:

    def __init__(self, label=None, id_=None, human="", synset=""):
        self.label = label
        self.id = id_
        self.human = human
        self.synset = synset


class BoundingBox:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @classmethod
    def from_points(cls, point_min, point_max):
        #Only supporting (x,y) points right now, add other formats and specifier if needed
        return cls(point_min[0], point_min[1], point_max[0], point_max[1])

    @classmethod
    def from_list(cls, coord_list, fmt='yxyx'):
        assert len(coord_list) == 4
        if fmt == 'xyxy':
            xmin = coord_list[0]
            ymin = coord_list[1]
            xmax = coord_list[2]
            ymax = coord_list[3]
        elif fmt == 'yxyx':
            xmin = coord_list[1]
            ymin = coord_list[0]
            xmax = coord_list[3]
            ymax = coord_list[2]
        elif fmt == 'xywh':
            xmin = coord_list[0]
            ymin = coord_list[1]
            xmax = xmin + coord_list[2]
            ymax = ymin + coord_list[3]
        elif fmt == 'yxhw':
            xmin = coord_list[1]
            ymin = coord_list[0]
            xmax = xmin + coord_list[3]
            ymax = ymin + coord_list[2]
        else:
            assert False, 'Unknown bbox list format'
        return cls(xmin, ymin, xmax, ymax)


class Polygon:

    def __init__(self, points=[]):
        self.points = points

    @classmethod
    def from_separates(cls, x_list, y_list):
        points = zip(x_list, y_list)
        return cls(points)

    def as_separates(self):
        return zip(*self.points)

    def clip(self, point_min, point_max):
        x_coords, y_coords = self.as_separates()
        x_coords = np.clip(np.asarray(x_coords), point_min[0], point_max[0])
        y_coords = np.clip(np.asarray(y_coords), point_min[1], point_max[1])
        self.points = list(zip(x_coords, y_coords))


class Record(object):
    def __init__(self):
        pass


    @classmethod
    def from_example(cls, example):
        record = Record()
        return record

    def to_example(self):
        example = tf.train.Example()
        return example


class RecordImage(Record):
    def __init__(self):
        self._objects = []


class ImgObject:

    def __init__(self):
        self.obj_class = ObjClass()
        self.bbox = None
        self.polygon = None
        self.mask = None


def _convert_image_objects(objects):
    feature_list = []
    for obj in objects:
        xmin = []
        ymin = []
        xmax = []
        ymax = []

        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    return feature_list

def _convert_image(filename, image_buffer, height, width, image_class=None, objects=[]):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """


    objects_feature_list = _convert_image_objects(objects)
    image_caption_list = []

    channels = 3
    colorspace = b'RGB'
    image_format = b'JPEG'

    image_features = tf.train.Features(feature={
        'img/h': _int64_feature(height),
        'img/w': _int64_feature(width),
        'img/ch': _int64_feature(channels),
        'img/col': _bytes_feature(colorspace),
        'img/fmt': _bytes_feature(image_format),
        'img/file': _bytes_feature(os.path.basename(filename).encode())
    }
    )

    # Add encoded image if we have it
    if image_buffer:
        image_features.feature['img/enc'].bytes_list.value.append(image_buffer)

    # Has label for whole image, vs just objects
    if image_class:
        image_features.feature['cls/label'].int64_list.value.append(image_class.label)
        image_features.feature['cls/syn'].bytes_list.value.append(image_class.synset.encode())
        image_features.feature['cls/txt'].bytes_list.value.append(image_class.human.encode())

    # Add image captions
    if has_captions:
        captions = []
        image_features.feature['capt'].bytes_list.value.extend(captions)

    if objects:

        label_list, bbox_list, poly_list, mask_list = _convert_objects(objects)

        if label_list:
            image_features.feature['obj/label'].int64_list.value.extend(label_list)

        if ymin:
            assert len(ymin) == len(ymax)
            assert len(ymin) == len(xmin)
            assert len(ymin) == len(xmax)

            image_features.feature['obj/bb/xmin'].int64_list.value.extend(xmin)
            image_features.feature['obj/bb/xmax'].int64_list.value.extend(xmax)
            image_features.feature['obj/bb/ymin'].int64_list.value.extend(ymin)
            image_features.feature['obj/bb/ymax'].int64_list.value.extend(ymax)

        if poly_len:
            # Multiple polygons encoded as
            #    poly_len - list of polygon lengths (number of x,y pairs for polygon) for each object
            #    poly_x - list of polygon x coordinates across all objects
            #    poly_y - list of polygon y coordinates across all objects
            image_features.feature['obj/poly/len'].int64_list.value.extend(poly_len)
            image_features.feature['obj/poly/x'].int64_list.value.extend(poly_x)
            image_features.feature['obj/poly/y'].int64_list.value.extend(poly_y)

        if masks:
            image_features.feature['obj/mask'].bytes_list.value.extend(masks)

    example = tf.train.Example(features=image_features)

    return example