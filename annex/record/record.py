# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

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
