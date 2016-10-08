# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# Based on original Work Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import xml.etree.ElementTree as ET

from .dataset import Dataset
from .helper import *
from record import BoundingBox, RecordImage


def _get_item(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def _get_int(name, root, index=0):
    return int(_get_item(name, root, index))


def _find_number_bounding_boxes(root):
    index = 0
    while True:
        if _get_int('xmin', root, index) == -1:
            break
        index += 1
    return index


class DatasetImagenet(Dataset):

    def __init__(
            self,
            name='train',
            data_dir='',
            labels_file='',
            labels_to_strings_file='',
            bbox_folder=''):
        super(DatasetImagenet, self).__init__(name=name, data_dir=data_dir)

        self._data_dir = data_dir
        self._labels_file = labels_file
        self._labels_to_strings_file = labels_to_strings_file
        self._bbox_folder = bbox_folder

        self._labels = []
        self._labels_to_human = {}
        self._bbox_map = {}

        # Load dataset annotations and metadata
        self._load_record_metadata()
        self._load_object_metadata()

    def _load_record_metadata(self):

        assert os.path.isfile(self._labels_file)
        assert os.path.isfile(self._labels_to_strings_file)

        try:
            self._labels = [l.strip() for l in open(self._labels_file, 'r').readlines()]
        except IOError:
            print("Error reading labels file %s" % self._labels_file)
            return

        try:
            lines = open(self._labels_to_strings_file, 'r').readlines()
            for l in lines:
                if l:
                    parts = l.strip().split('\t')
                    assert len(parts) == 2
                    label = parts[0]
                    human = parts[1]
                    self._labels_to_human[label] = human
        except IOError:
            print("Error reading label to string mapping file %s" % self._labels_to_strings_file)

        filenames, label_texts, label_ids, _ = find_data_files_and_labels_shallow(
            self._data_dir,
            self._labels,
            types=('.jpg', '.jpeg'),
            include_empty_labels=True,
            add_background_label=True
        )

        for filename, label_text, label_id in zip(filenames, label_texts, label_ids):
            assert label_id in self._labels_to_human, ('Failed to find: %s' % label)
            human_text = self._labels_to_human[label]
            self._records[filename] = {
                'label_id': label_id,
                'label_text': label_text,
                'human_text': human_text
            }

    def _load_object_metadata(self):
        assert os.path.isdir(self._bbox_folder)
        self._load_xml_bounding_boxes(self._bbox_folder, self._labels)

    def _load_xml_bounding_boxes(self, bbox_folder, labels=None, match_image_label=False):

        xml_files, label_texts, _, _ = find_data_files_and_labels_shallow(
            bbox_folder, labels, types=('.xml'), full_file_path=True)
        print('Identified %d XML files in %s' % (len(xml_files), bbox_folder), file=sys.stderr)

        skipped_boxes = 0
        skipped_files = 0
        saved_boxes = 0
        saved_files = 0
        for file_index, filename in enumerate(xml_files):
            label = label_texts[file_index]
            bboxes = self._process_xml_annotation(filename)
            assert bboxes is not None, 'No bounding boxes found in ' + filename

            found_box = False
            for bbox in bboxes:
                if labels:
                    if match_image_label and bbox.label != label and bbox.label in labels:
                        # Skip invalid sysnets as there are some in annotations.
                        skipped_boxes += 1
                        continue

                if (bbox.xmin_scaled >= bbox.xmax_scaled or bbox.ymin_scaled >= bbox.ymax_scaled):
                    skipped_boxes += 1
                    continue

                # bbox.filename occasionally contains '%s' in the name. Fix by using the basename of the XML file.
                image_filename = os.path.splitext(os.path.basename(filename))[0]
                image_filename = os.path.join(image_filename, '.jpeg')
                self._bbox_map[image_filename] = bbox

                saved_boxes += 1
                found_box = True

            if found_box:
                saved_files += 1
            else:
                skipped_files += 1

            if not file_index % 5000:
                print('... processed %d of %d XML files.' % (file_index + 1, len(xml_files)), file=sys.stderr)
                print('... skipped %d boxes and %d XML files.' % (skipped_boxes, skipped_files), file=sys.stderr)

        print('Finished processing %d XML files.' % len(xml_files), file=sys.stderr)
        print('Skipped %d XML files.' % skipped_files, file=sys.stderr)
        print('Skipped %d bounding boxes.' % skipped_boxes, file=sys.stderr)
        print('Created %d bounding boxes from %d annotated images.' % (saved_boxes, saved_files), file=sys.stderr)

    def _process_xml_annotation(self, xml_file):
        # pylint: disable=broad-except
        try:
            tree = ET.parse(xml_file)
        except Exception:
            print('Failed to parse: ' + xml_file, file=sys.stderr)
            return None
        # pylint: enable=broad-except
        root = tree.getroot()

        num_boxes = _find_number_bounding_boxes(root)
        boxes = []

        for index in range(num_boxes):
            # convert to 0-based
            bbox = BoundingBox().from_xyxy(
                _get_int('xmin', root, index),
                _get_int('ymin', root, index),
                _get_int('xmax', root, index),
                _get_int('ymax', root, index),
            )

            # This width/height is the dimension of the image used to create the bbox
            # annotation. Used for scaling of bbox to actual image width/height.
            bbox_image_width = _get_int('width', root)
            bbox_image_height = _get_int('height', root)
            # convert to floating point relative values [0.0, 1.0]
            bbox.to_relative(bbox_image_width, bbox_image_height)

            image_filename = _get_item('filename', root) + '.JPEG'
            image_label = _get_item('name', root)

            # Some images contain bounding box annotations that
            # extend outside of the supplied image. See, e.g.
            # n03127925/n03127925_147.xml
            # Additionally, for some bounding boxes, the min > max
            # or the box is entirely outside of the image.
            bbox.clip(1.0, 1.0, 1.0, 1.0)

            boxes.append((image_filename, image_label, bbox))

        return boxes

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

        return record

    def is_cmyk(self, filename):
        match_list = [
            'n01739381_1309.JPEG', 'n02077923_14822.JPEG',
            'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
            'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
            'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
            'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
            'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
            'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
            'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
            'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
            'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
            'n07583066_647.JPEG', 'n13037406_4650.JPEG']
        return filename.split('/')[-1] in match_list

    def is_png(self, filename):
        return 'n02105855_2933.JPEG' in filename

    def records_as_list(self, start=0, end=None, include_objects=False):
        return list(self.generate_records(start, end, include_objects))

    def generate_records(self, start=0, end=None, include_objects=False):
        for rec_id in self._records_index[start:end]:
            record = self._rec_dict_to_class(self._records[rec_id], include_objects)
            yield record
