# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

import os
import tensorflow as tf
from abc import abstractmethod

FLAGS = tf.app.flags.FLAGS


class Dataset(object):

    def __init__(self, name, data_dir=''):

        self.name = name

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self._records = {}
        self._records_index = []
