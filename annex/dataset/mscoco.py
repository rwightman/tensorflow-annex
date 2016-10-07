from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
import pandas as pd
from collections import defaultdict

from external import CocoData
from record import ClassLabel, RecordImage, ImageObject, BoundingBox, Polygon2D, MaskRle
from record.example import *
from .dataset import Dataset


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('annotation_file', '', 'Annotation file')

default_encode_params = {
    'shuffle': False,
    'background_class': False,
    'filter_categories': [],
    'scale_percent': 100,
    'scale_largest': 0,
    'include_objects': True,
}


class DatasetCoco(Dataset):

    def __init__(
            self,
            name='train',
            data_dir='',
            annotation_file=''):
        super(DatasetCoco, self).__init__(name=name, data_dir=data_dir)

        if not annotation_file:
            annotation_file = FLAGS.annotation_file
        assert os.path.isfile(annotation_file)

        print('Loading Coco annotations from %s...' % annotation_file)
        self._cocod = CocoData(annotation_file=annotation_file)
        self._classes = {}
        self._objects = {}
        self._record_to_objects = defaultdict(list)

        #FIXME make param
        self._shuffle_records = False
        #self._shuffle_objects = False
        self._background_class = True
        self._filter_cats = []

        #TODO
        # filter %of total dataset per category
        # scaling / color conversion params for processor

        # Load dataset annotations and metadata
        self._load_class_metadata()
        self._load_record_metadata()
        self._load_object_metadata()

    def _load_class_metadata(self):
        label_index = 1 if self._background_class else 0
        cat_ids = sorted(self._cocod.get_cat_ids(cat_ids=self._filter_cats))
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

        if self._filter_cats:
            self._records = self._cocod.get_images_dict_by_id(
                self._cocod.get_image_ids(cat_ids=self._filter_cats))
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

    def _rec_dict_to_class(self, recd, include_objects=False):
        if not isinstance(recd, dict):
            rec_id = recd
            recd = self._records[rec_id]
        else:
            rec_id = recd['id']

        record = RecordImage(
            rec_id=rec_id,
            filename=recd['file_name'],
            height=recd['height'],
            width=recd['width'],
        )

        if include_objects:
            #rec_objs = self._record_to_objects[rec_id]
            objects = [
                self._obj_dict_to_class(self._objects[obj_id], record.width, record.height)
                for obj_id in self._record_to_objects[rec_id]]
            record.add_objects(objects)

        return record

    def _obj_dict_to_class(self, objd, w, h):
        class_id = objd['category_id']
        bbox = BoundingBox.from_list(objd['bbox'], fmt='xywh')
        polygons = []
        mask = None
        if objd['segmentation']:
            if objd['iscrowd']:
                mask = MaskRle.from_dict(src_dict=objd['segmentation'])
            else:
                mask = MaskRle.from_list(w, h, objd['segmentation'])
                polygons = [Polygon2D.from_list(l) for l in objd['segmentation']]

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

    def records_as_list(self, start=0, end=None, include_objects=False):
        return list(self.generate_records(start, end, include_objects))

    def generate_records(self, start=0, end=None, include_objects=False):
        for rec_id in self._records_index[start:end]:
            record = self._rec_dict_to_class(self._records[rec_id], include_objects)
            yield record
