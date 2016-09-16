from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .dataset import Dataset


class DatasetCoco(Dataset):

    def __init__(self):
        pass

    def map_objects(self):
        #bbox, polygon/mask, obj labels from big JSON
        pass

    def map_labels(self):
        # no per image label, only objects
        # per image hxw and other metadata tags
        # has per image captions
        pass