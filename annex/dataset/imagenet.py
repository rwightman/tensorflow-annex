from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .dataset import Dataset
from .helper import *


def _process_dataset(name, dir, num_shards, metadata):

    filenames, synsets, labels, _ = find_data_files_and_labels_shallow(directory, FLAGS.labels_file)
    human_text = human_readable_labels(synsets, synset_to_human)
    bboxes = image_bounding_boxes(filenames, image_to_bboxes)

    return

class DatasetImagenet(Dataset):

    def __init__(self, data_dir='.'):
        self._data_dir = data_dir


    def map_objects(self):
        #bbox from XML files
        obj = {}
        xf = open(xml_file)
        xml_tree = parse(xf)
        for entry in xml_tree:
            obj[id] = entry
        pass


    def map_labels(self, labels_file):

        self._labels = load_labels(labels_file)
        self._human = load_human(human_label_map)

        self._files, _, _, _ = find_data_files_and_labels_shallow(
            self._data_dir,
            self._labels,
            include_empty_labels=True,
            add_background_label=True
        )
        #sysnet id -> human text
        #label id (sysnet) from file folder...
        #valid sysnets?