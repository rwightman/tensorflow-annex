# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

import os
import random
import math
import tensorflow as tf
from collections import defaultdict

from external import CocoData
from record import ClassLabel, RecordImage, ImageObject, BoundingBox, Polygon2D, MaskRle
from record.example import *
from .dataset import Dataset


default_encode_params = {
    'shuffle': False,
    'background_class': True,
    'filter_categories': [],
    'scale_percent': 100,
    'scale_longest': 0,
    'include_objects': True,
}


class DatasetCoco(Dataset):

    def __init__(
            self,
            name='train',
            data_dir='',
            annotation_file='',
            encode_params=None):
        super(DatasetCoco, self).__init__(name=name, data_dir=data_dir)

        assert os.path.isfile(annotation_file)

        print('Loading Coco annotations from %s...' % annotation_file)
        self._cocod = CocoData(annotation_file=annotation_file)
        self._classes = {}
        self._objects = {}
        self._record_to_objects = defaultdict(list)

        # Params for encoding records
        params = default_encode_params
        if encode_params:
            params.update(encode_params)
        self._shuffle_records = params['shuffle']
        self._background_class = params['background_class']
        self._filter_categories = params['filter_categories']
        self._include_objects = params['include_objects']
        self._scale_percent = params['scale_percent']
        self._scale_longest = params['scale_longest']
        # it only makes sense to specify a percent scaling OR scale to longest edge, not both
        assert not self._scale_longest or self._scale_percent == 100

        #TODO
        # filter %of total dataset per category
        # scaling / color conversion params for processor

        # Load dataset annotations and metadata
        self._load_class_metadata()
        self._load_record_metadata()
        self._load_object_metadata()

    def _load_class_metadata(self):
        label_index = 1 if self._background_class else 0
        cat_ids = sorted(self._cocod.get_cat_ids(cat_ids=self._filter_categories))
        self._labels = []
        for cat_id in cat_ids:
            catd = self._cocod.cats[cat_id]
            obj_class = self._cat_dict_to_class(catd, label_index)
            self._classes[cat_id] = obj_class
            self._labels.append(obj_class)
            label_index += 1

    def _load_record_metadata(self):
        # no per image label, only objects
        # per image hxw and other metadata tags
        # has per image captions

        if self._filter_categories:
            self._records = self._cocod.get_images_dict_by_id(
                self._cocod.get_image_ids(cat_ids=self._filter_categories))
        else:
            self._records = self._cocod.images

        if self._shuffle_records:
            self._records_index = random.shuffle(self._records.keys())
        else:
            self._records_index = sorted(self._records.keys())

    def _load_object_metadata(self):
        for rec_id in self._records_index:
            anns = {ann['id']: ann for ann in self._cocod.image_to_anns[rec_id]}
            self._record_to_objects[rec_id].extend(sorted(list(anns.keys())))
            self._objects.update(anns)

    def _cat_dict_to_class(self, catd, label_index):
        assert isinstance(catd, dict)
        class_id = catd['id']
        obj_class = ClassLabel(
            label=label_index,
            class_id=class_id,
            text=catd['name'])
        return obj_class

    def _rec_dict_to_class(self, recd):
        if not isinstance(recd, dict):
            rec_id = recd
            recd = self._records[rec_id]
        else:
            rec_id = recd['id']

        height = recd['height']
        width = recd['width']
        scaled = False
        if self._scale_percent != 100:
            height = (height * self._scale_percent) // 100
            width = (width * self._scale_percent) // 100
            scaled = True
        elif self._scale_longest:
            if height > width:
                height = self._scale_longest
                ratio = width / height
                width = math.floor(ratio * self._scale_longest)
            else:
                width = self._scale_longest
                ratio = height / width
                height = math.floor(ratio * self._scale_longest)
            scaled = True

        record = RecordImage(
            rec_id=rec_id,
            filename=recd['file_name'],
            height=height,
            width=width,
        )
        if scaled:
            record.height_orig = recd['height']
            record.width_orig = recd['width']

        if self._include_objects:
            #rec_objs = self._record_to_objects[rec_id]
            objects = [
                self._obj_dict_to_class(self._objects[obj_id], record)
                for obj_id in self._record_to_objects[rec_id]]
            record.add_objects(objects)

        return record

    def _obj_dict_to_class(self, objd, record):
        class_id = objd['category_id']
        bbox = BoundingBox.from_list(objd['bbox'], fmt='xywh')
        polygons = []
        mask = None
        if objd['segmentation']:
            if objd['iscrowd']:
                mask = MaskRle.from_dict(src_dict=objd['segmentation'])
            else:
                mask = MaskRle.from_list(record.width, record.height, objd['segmentation'])
                polygons = [Polygon2D.from_list(l) for l in objd['segmentation']]
        #FIXME impact of scaling?
        obj = ImageObject(
            obj_id=objd['id'],
            obj_class=self._classes[class_id],  # cached classes
            bbox=bbox,
            polygons=polygons,
            mask=mask
        )
        obj.is_crowd = objd['iscrowd']

        return obj

    def is_cmyk(self, filename):
        return False

    def is_png(self, filename):
        return False

    def num_records(self):
        return len(self._records_index)

    def records_as_list(self, start=0, end=None):
        return list(self.generate_records(start, end))

    def generate_records(self, start=0, end=None):
        for rec_id in self._records_index[start:end]:
            record = self._rec_dict_to_class(self._records[rec_id])
            yield record
