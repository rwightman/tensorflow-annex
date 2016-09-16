from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




class Dataset(object):

    def __init__(self):

        if True:
            self._data_files = {'train': [], 'validation': []}
        else:
            self._data_files = {'test': []}

        self._items = {}
        self._objects = {}

        pass

    def map_objects(self):
        pass


    def map_items(self):
        pass