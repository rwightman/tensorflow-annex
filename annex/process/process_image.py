from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from .process import Processor


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
        return image


###

class ProcessorImage(Processor):
    def __init__(self):
        super(ProcessorImage, self).__init__()
        self._coder = ImageCoder()
        self._check_images = False
        self._skip_example = False
        self._out_of_band_images = False  # encoded images in Record/example vs filesystem
        self._out_of_band_dest = ''

    def _process_record(self, record, writer):
        """Process a single image record.

        Args:
          filename: string, path to an image file e.g., '/path/to/example.JPG'.
          coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
          image_buffer: string, JPEG encoding of RGB image.
          height: integer, image height in pixels.
          width: integer, image width in pixels.
        """
        # Read the image file.
        filename = record.filename
        image_data = tf.gfile.FastGFile(filename, 'rb').read()

        # Clean the dirty data.
        if self.dataset.is_png(filename):
            # 1 image is a PNG.
            print('Converting PNG to JPEG for %s' % filename)
            image_data = self.coder.png_to_jpeg(image_data)
        if self.dataset.is_cmyk(filename):
            # 22 JPEG images are in CMYK colorspace.
            print('Converting CMYK to RGB for %s' % filename)
            image_data = self.coder.cmyk_to_rgb(image_data)

        if self._check_images:
            # Decode the RGB JPEG and check that image converted to RGB
            image = self.coder.decode_jpeg(image_data)
            assert len(image.shape) == 3
            height = image.shape[0]
            width = image.shape[1]
            assert image.shape[2] == 3

        if self._out_of_band_images:
            out_of_band_filename = os.path.join(self._out_of_band_dest, filename)
            tf.gfile.FastGFile(out_of_band_filename, 'wb').write(image_data)
        else:
            record.set_encoded(image_data)







