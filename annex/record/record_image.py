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
from typing import Iterable
from .record import Record
from .geometry_2d import BoundingBox, Polygon2D, MaskRle
from .class_label import ClassLabel, class_label_lists_to_features
from .example import *


class ImageObject(object):

    def __init__(
            self,
            obj_id,
            obj_class: ClassLabel = None,
            bbox: BoundingBox = None,
            polygons: list = [],
            mask: MaskRle = None):

        self.obj_id = obj_id
        self.obj_class = obj_class
        self.bbox = bbox
        self.polygons = polygons
        self.mask = mask
        self.is_crowd = False


class RecordImage(Record):

    def __init__(
            self,
            rec_id: int,
            filename: str,
            image_class: ClassLabel = None,
            height: int = None,
            width: int = None,
            chan: int = 3,
            colorspace: str = 'RGB',
            encode_format: str = 'JPEG'):

        super(RecordImage, self).__init__(rec_id=rec_id, filename=filename)
        self.image_class = image_class  # image's object class, some datasets won't have record level classes and labels
        assert (height and height > 0) or (width and width > 0)
        self.height = height if not height else width
        self.width = height if not width else width
        self.height_orig = None  # only set if image to be rescaled
        self.width_orig = None  # only set if image to be rescaled
        self.chan = chan
        self.colorspace = colorspace
        self.encode_format = encode_format
        self.encoded = bytes()
        self.captions = []
        self.objects = []
        self.objects_have_bbox = False
        self.objects_have_polygons = False
        self.objects_have_masks = False

    @classmethod
    def from_example(cls, example):
        record = RecordImage()
        return record

    def add_objects(self, objects: Iterable[ImageObject]):
        for o in objects:
            assert o.obj_class
            if o.bbox:
                self.objects_have_bbox = True
            if o.polygons:
                self.objects_have_polygons = True
            if o.mask:
                self.objects_have_masks = True
            self.objects.append(o)

    def set_encoded(self, encoded_bytes):
        assert len(encoded_bytes)
        if not isinstance(encoded_bytes, bytes):
            encoded_bytes = bytes(encoded_bytes)
        self.encode_format = encoded_bytes

    def _objects_to_features(self, feature_dict):

        # For each object there will be one entry in each of the below lists, if the object
        # has no instance of that type there will be a null/empty entry. All below lists
        # must have the same length.

        class_label_list = []
        class_id_list = []
        class_text_list = []
        class_id_kind = FeatureKind.INT64
        bbox_xmin_list = []
        bbox_ymin_list = []
        bbox_width_list = []
        bbox_height_list = []
        bbox_kind = FeatureKind.FLOAT
        polygon_counts_list = []  # number of polygons and polygon co-ordinate list lengths per object
        polygon_x_list = []
        polygon_y_list = []
        polygon_kind = FeatureKind.FLOAT
        mask_list = []

        for obj in self.objects:

            assert obj.obj_class, 'Objects must have a class label if they are present'
            if obj.obj_class:
                class_label_list.append(obj.obj_class.label)
                if obj.obj_class.id is not None:
                    class_id_list.append(obj.obj_class.id)
                if obj.obj_class.text is not None:
                    class_text_list.append(obj.obj_class.text)

            if self.objects_have_bbox:
                if obj.bbox:
                    obj.bbox.append_to_lists(
                        bbox_xmin_list, bbox_ymin_list, bbox_width_list, bbox_height_list)
                else:
                    BoundingBox().append_to_lists(bbox_xmin_list, bbox_ymin_list, bbox_width_list, bbox_height_list)

            if self.objects_have_polygons:
                counts = [len(obj.polygons)]
                for p in obj.polygons:
                    count = p.append_to_lists(polygon_x_list, polygon_y_list, delta=True)
                    counts.append(count)
                assert len(counts) == len(obj.polygons) + 1
                polygon_counts_list.extend(counts)

            if self.objects_have_masks:
                if obj.mask:
                    obj.mask.append_to_list(mask_list)
                else:
                    mask_list.append([])

        assert not class_label_list or len(class_label_list) == len(self.objects)
        assert not bbox_xmin_list or len(bbox_xmin_list) == len(self.objects)  # no bounding boxes or one per object
        assert not mask_list or len(mask_list) == len(self.objects)  # no masks or one per object

        class_label_lists_to_features(
            feature_dict, class_label_list, class_id_list, class_text_list, class_id_kind, key_prefix='obj/')

        if bbox_xmin_list:
            assert len(bbox_xmin_list) == len(bbox_ymin_list) == len(bbox_width_list) == len(bbox_height_list)
            feature_dict['obj/bb/xmin'] = feature_select(bbox_xmin_list, bbox_kind)
            feature_dict['obj/bb/ymin'] = feature_select(bbox_ymin_list, bbox_kind)
            feature_dict['obj/bb/w'] = feature_select(bbox_width_list, bbox_kind)
            feature_dict['obj/bb/h'] = feature_select(bbox_height_list, bbox_kind)

        if polygon_counts_list:
            assert len(polygon_x_list) == len(polygon_y_list)
            # Multiple polygons encoded as
            #    poly_len - list of polygon lengths (number of x,y pairs for polygon) for each object
            #    poly_x - list of polygon x coordinates across all objects
            #    poly_y - list of polygon y coordinates across all objects
            feature_dict['obj/poly/counts'] = feature_int64(polygon_counts_list)
            feature_dict['obj/poly/x'] = feature_select(polygon_x_list, polygon_kind)
            feature_dict['obj/poly/y'] = feature_select(polygon_y_list, polygon_kind)

        if mask_list:
            feature_dict['obj/mask'] = feature_bytes_list(mask_list)

    def to_example(self):

        feature_dict = {
            'img/h': feature_int64(self.height),
            'img/w': feature_int64(self.width),
            'img/ch': feature_int64(self.chan),
            'img/col': feature_bytes(self.colorspace),
            'img/fmt': feature_bytes(self.encode_format),
            'img/file': feature_bytes(os.path.basename(self.filename))
        }

        if self.encoded:
            feature_dict['img/enc'] = feature_bytes(self.encoded)

        if self.image_class:
            self.image_class.to_feature(feature_dict, 'img/')

        if self.captions:
            feature_dict['img/capt'] = feature_bytes_list(self.captions)

        if self.objects:
            self._objects_to_features(feature_dict)

        image_features = tf.train.Features(feature=feature_dict)
        example = tf.train.Example(features=image_features)
        return example
