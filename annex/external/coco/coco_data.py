# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  get_ann_ids  - Get ann ids that satisfy given filter conditions.
#  get_cat_ids  - Get cat ids that satisfy given filter conditions.
#  get_image_ids  - Get img ids that satisfy given filter conditions.
#  get_anns_by_id   - Load anns with the specified ids.
#  get_cats_by_id   - Load cats with the specified ids.
#  get_images_by_id   - Load images with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  show_anns   - Display the specified annotations.
#  load_results    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>get_ann_ids, COCO>get_cat_ids,
# COCO>get_image_ids, COCO>get_anns_by_id, COCO>get_cats_by_id,
# COCO>get_images_by_id, COCO>segToMask, COCO>show_anns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import urllib
import copy
import itertools
import os
from collections import defaultdict
from ..mask import mask

__author__ = 'tylin'
__version__ = '2.0'


class CocoData:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param str annotation_file: location of annotation file
        :param str image_folder: location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = dict()
        self.anns = dict()
        self.cats = dict()
        self.images = dict()
        self.image_to_anns = defaultdict(list)
        self.cat_to_images = defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, "annotation file format %s not supported" % (type(dataset))
            print('Done (t=%0.2fs)' % (time.time() - tic))
            self.dataset = dataset
            self.create_index()

    def create_index(self):
        # create index
        print('creating index...')
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.image_to_anns[ann['image_id']].append(ann)
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for image in self.dataset['images']:
                self.images[image['id']] = image

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat
            for ann in self.anns.values():
                self.cat_to_images[ann['category_id']].append(ann['image_id'])

        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s' % (key, value))

    def get_ann_ids(self, image_ids=[], cat_ids=[], area_range=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param: image_ids  (int array)   : get anns for given images
               cat_ids  (int array)     : get anns for given cats
               area_range (float array) : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)        : get anns for given crowd label (False or True)
        :return: ids (int array)        : integer array of ann ids
        """
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]

        if image_ids:
            lists = [self.image_to_anns[image_id] for image_id in image_ids if image_id in self.image_to_anns]
            anns = list(itertools.chain.from_iterable(lists))
        else:
            anns = self.anns.values()

        if cat_ids:
            anns = [ann for ann in anns if ann['category_id'] in cat_ids]

        if area_range:
            anns = [ann for ann in anns if ann['area'] > area_range[0] and ann['area'] < area_range[1]]

        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def get_cat_ids(self, cat_names=[], supcat_names=[], cat_ids=[]):
        """
        filtering parameters. default skips that filter.
        :param: cat_names (str array)  : get cats for given cat names
        :param: supcat_names (str array)  : get cats for given supercategory names
        :param: cat_ids (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_names = cat_names if type(cat_names) == list else [cat_names]
        supcat_names = supcat_names if type(supcat_names) == list else [supcat_names]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]

        cats = self.cats
        if cat_names:
            cats = {k: v for (k, v) in cats.items() if v['name'] in cat_names}
        if supcat_names:
            cats = {k: v for (k, v) in cats.items() if v['supercategory'] in supcat_names}
        if cat_ids:
            ids = [cat for cat in cats.keys() if cat in cat_ids]
        else:
            ids = [cat for cat in cats.keys()]
        return ids

    def get_image_ids(self, image_ids=[], cat_ids=[], cat_union=False):
        """
        Get img ids that satisfy given filter conditions.
        :param list[int] image_ids: get images for given ids
        :param list[int] cat_ids: get images with all given cats
        :return list[int] ids: integer array of img ids
        """
        image_ids = set(image_ids) if type(image_ids) == list else {image_ids}
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]

        if not cat_ids:
            if image_ids:
                ids = image_ids & set(self.images.keys())
            else:
                ids = self.images.keys()
        else:
            ids = set()
            for cat_id in cat_ids:
                update_ids = set(self.cat_to_images[cat_id])
                if image_ids:
                    update_ids &= image_ids
                ids = ids | update_ids if cat_union else ids & update_ids
        return list(ids)

    def get_anns_by_id(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[i] for i in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def get_anns_dict_by_id(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return {i: self.anns[i] for i in ids}
        elif type(ids) == int:
            return {ids: self.anns[ids]}

    def get_cats_by_id(self, ids=[]):
        """
        Load cats with the specified ids.
        :param: ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[i] for i in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def get_images_by_id(self, ids=[]):
        """
        Load anns with the specified ids.
        :param int[] ids: integer ids specifying img
        :return images: (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.images[i] for i in ids]
        elif type(ids) == int:
            return [self.images[ids]]

    def get_images_dict_by_id(self, ids=[]):
        """
        Load anns with the specified ids.
        :param int[] ids: integer ids specifying img
        :return images: (object array) : loaded img objects
        """
        if type(ids) == list:
            return {i: self.images[i] for i in ids}
        elif type(ids) == int:
            return {ids: self.images[ids]}

    def show_anns(self, anns):
        """
        Display the specified annotations.
        :param anns: (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception("datasetType not supported")
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((len(seg) / 2, 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.images[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = mask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = mask.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        else:  # ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.get_cats_by_id(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k',
                             markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=2)
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def load_results(self, results_file):
        """
        Load result file and return a result api object.
        :param   results_file (str)     : file name of result file
        :return: results (obj)         : result api object
        """
        results = CocoData()
        results.dataset['images'] = [image for image in self.dataset['images']]

        print('Loading and preparing results...     ')
        tic = time.time()
        if type(results_file) == str:
            anns = json.load(open(results_file))
        elif type(results_file) == np.ndarray:
            anns = self.load_numpy_annotations(results_file)
        else:
            anns = results_file
        assert type(anns) == list, 'results in not an array of objects'
        anns_img_ids = [ann['image_id'] for ann in anns]
        assert set(anns_img_ids) == (set(anns_img_ids) & set(self.get_image_ids())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            image_ids = set([img['id'] for img in results.dataset['images']]) & set([ann['image_id'] for ann in anns])
            results.dataset['images'] = [img for img in results.dataset['images'] if img['id'] in image_ids]
            for id_, ann in enumerate(anns):
                ann['id'] = id_ + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id_, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if 'segmentation' not in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id_ + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id_, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = mask.area([ann['segmentation']])[0]
                if 'bbox' not in ann:
                    ann['bbox'] = mask.toBbox([ann['segmentation']])[0]
                ann['id'] = id_ + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id_, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id_ + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        print('DONE (t=%0.2fs)' % (time.time() - tic))

        results.dataset['annotations'] = anns
        results.create_index()
        return results

    @staticmethod
    def load_numpy_annotations(data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print("Converting ndarray to lists...")
        assert type(data) == np.ndarray
        print(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print("%d/%d" % (i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann
