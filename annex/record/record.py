from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod


class Record(object):
    def __init__(self, rec_id, filename):
        self.rec_id = rec_id   # unique id of record in dataset
        self.filename = filename   # filename for record data

    @classmethod
    def from_example(cls, example):
        record = Record()
        return record

    @abstractmethod
    def to_example(self):
        #implemented in derived classes
        pass
