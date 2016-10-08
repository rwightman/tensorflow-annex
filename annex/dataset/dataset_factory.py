# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

import os

from .imagenet import DatasetImagenet
from .mscoco import DatasetCoco

_datasets = {
    "imagenet": DatasetImagenet,
    "mscoco": DatasetCoco,
}


def is_valid_type(type):
    return type in _datasets


def create(type, *args, **kwargs):
    if type not in _datasets:
        print("Unknown dataset type!")
        return None  #FIXME throw?
    return _datasets[type](*args, **kwargs)