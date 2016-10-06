from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
#import tensorflow as tf
from collections import Counter


def find_data_files_and_labels_deep(
        data_dir,
        types=('.jpg', '.jpeg', '.png'),
        full_file_path=False,
        add_background_label=False):

    label_counts = Counter()
    label_ids = []
    label_texts = []
    filenames = []

    label_index = 1 if add_background_label else 0

    for root, _, files in os.walk(data_dir, topdown=False):
        label = os.path.relpath(root, data_dir) if (root != data_dir) else ''
        matching_files = []
        matching_count = 0
        for f in files:
            if os.path.splitext(f)[1].lower() in types:
                f = os.path.join(root if full_file_path else label, f)
                matching_files.append(f)
                matching_count += 1
        if matching_count:
            if label_index and not label_index % 100:
                print('Finished finding files in %d of %d classes.' % (label_index, len(unique_labels)))
            label_counts[label] += matching_count
            label_ids.extend([label_index] * matching_count)
            label_texts.extend([label] * matching_count)
            filenames.extend(matching_files)
            label_index += 1

    print('Found %d data files across %d labels inside %s.' % (len(filenames), len(label_counts), data_dir))

    return filenames, label_texts, label_ids, label_counts


def find_data_files_and_labels_shallow(
        data_dir,
        unique_labels=[],
        types=('.jpg', '.jpeg', '.png'),
        full_file_path=False,
        include_empty_labels=False,
        add_background_label=False):

    label_counts = Counter()
    label_ids = []
    label_texts = []
    filenames = []

    # if labels are not specified as argument, we will find them ourselves
    if not unique_labels:
        unique_labels = next(os.walk(data_dir))[1]

    label_index = 1 if add_background_label else 0

    for label in unique_labels:
        label_path = os.path.join(data_dir, label)
        files = os.listdir(label_path)
        matching_files = []
        matching_count = 0

        for f in files:
            full_f = os.path.join(label_path, f)
            if os.path.isfile(full_f) and os.path.splitext(f)[1].lower() in types:
                f = full_f if full_file_path else os.path.join(label, f)
                matching_files.append(f)
                matching_count += 1

        if include_empty_labels or matching_count:
            if label_index and not label_index % 100:
                print('Finished finding files in %d of %d classes.' % (label_index, len(unique_labels)))
            label_counts[label] += matching_count
            label_ids.extend([label_index] * matching_count)
            label_texts.extend([label] * matching_count)
            filenames.extend(matching_files)
            label_index += 1

    print('Found %d data files across %d labels inside %s.' % (len(filenames), len(label_counts), data_dir))

    return filenames, label_texts, label_ids, label_counts


def find_data_files_shallow(data_dir, unique_labels=[], types=('.jpg', 'jpeg'), full_file_path=False):
    return find_data_files_and_labels_shallow(
        data_dir, unique_labels, types,
        full_file_path=full_file_path,
        include_empty_labels=False,
        add_background_label=False)[0]


def load_labels(labels_file):
    try:
        unique_labels = [l.strip() for l in open(labels_file, 'r').readlines()]
    except OSError:
        return []
    return unique_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    args = parser.parse_args()
    data_dir = args.data_dir

    files, labels, ids, counts = find_data_files_and_labels_deep(data_dir)

    files2, labels2, ids2, counts2 = find_data_files_and_labels_shallow(data_dir)


if __name__ == '__main__':
    main()