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