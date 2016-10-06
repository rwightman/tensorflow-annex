import numbers
from .example import *


class ClassLabel:

    def __init__(self,
                 label: int,
                 class_id=None,
                 text: str = None):

        assert isinstance(label, numbers.Integral)
        self.label = label  # integer class label, dataset instance specific
        self.id = class_id  # unique class identifier, numeric or string
        self.text = text  # brief human understandable text string

    def to_feature(self, features, prefix=''):
        features[prefix + '/cls/lbl'] = feature_int64(self.label)
        if self.id is not None:
            kind = FeatureKind.INT64
            if not isinstance(self.id, numbers.Integral):
                kind = FeatureKind.BYTES
            features[prefix + '/cls/id'] = feature_select(self.id, kind)
        if self.text is not None:
            features[prefix + '/cls/txt'] = feature_bytes(self.text)


def class_label_list_to_features(features, class_label_list, key_prefix=''):
    label_list = []
    id_list = []
    text_list = []
    id_kind = FeatureKind.INT64
    for x in class_label_list:
        assert isinstance(x, ClassLabel)
        label_list.append(x.label)
        if x.id is not None:
            id_list.append(x.id)
            if not isinstance(x.id, numbers.Integral):
                id_kind = FeatureKind.BYTES_LIST
        if x.text is not None:
            text_list.append(x.text)
    return class_label_lists_to_features(features, label_list, id_list, text_list, id_kind, key_prefix=key_prefix)


def class_label_lists_to_features(features, label_list, id_list, text_list, id_kind: FeatureKind, key_prefix=''):
    assert len(label_list)
    assert not len(id_list) or len(id_list) == len(label_list)
    assert not len(text_list) or len(text_list) == len(label_list)
    features[key_prefix + '/cls/lbl'] = feature_int64(label_list)
    features[key_prefix + '/cls/id'] = feature_select(id_list, id_kind)
    features[key_prefix + '/cls/txt'] = feature_bytes_list(text_list)