import numpy as np
import datetime
import time
import copy
from collections import defaultdict
from .. import mask

__author__ = 'tsungyi'


class CocoEval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  coco_gt=..., coco_dt=...       # load dataset and results
    #  E = CocoEval(coco_gt,coco_dt); # initialize CocoEval object
    #  E.params.recall_thresholds = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  image_ids     - [all] N img ids to use for evaluation
    #  cat_ids     - [all] K cat ids to use for evaluation
    #  iou_thresholds    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recall_thresholds    - [0:.01:1] R=101 recall thresholds for evaluation
    #  area_range    - [...] A=4 object area ranges for evaluation
    #  max_dets    - [1 10 100] M=3 thresholds on max detections per image
    #  iou_type    - ['segm'] set iou_type to 'segm', 'bbox' or 'keypoints'
    #  iou_type replaced the now DEPRECATED use_segmentation parameter.
    #  use_cats    - [1] if true use category labels for evaluation
    # Note: if use_cats=0 category labels are ignored as in proposal scoring.
    # Note: multiple area_ranges [Ax2] and max_dets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "eval_images" with fields:
    #  dt_ids      - [1xD] id for each of the D detections (dt)
    #  gt_ids      - [1xG] id for each of the G ground truths (gt)
    #  dt_matches  - [TxD] matching gt id at each IoU or 0
    #  gt_matches  - [TxG] matching dt id at each IoU or 0
    #  dt_scores   - [1xD] confidence of each dt
    #  gt_ignore   - [1xG] ignore flag for each gt
    #  dt_ignore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "eval_images" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, coco_gt=None, coco_dt=None, iou_type="segm"):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param coco_gt: coco object with ground truth annotations
        :param coco_dt: coco object with detection results
        :return: None
        '''
        if not iou_type:
            print("iou_type not specified. use default iou_type segm")
        self.coco_gt = coco_gt  # ground truth COCO API
        self.coco_dt = coco_dt  # detections COCO API
        self.params = {}  # evaluation parameters
        self.eval_images = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self._params_eval = None  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if coco_gt is not None:
            self.params.image_ids = sorted(coco_gt.get_image_ids())
            self.params.cat_ids = sorted(coco_gt.get_cat_ids())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''

        def _to_mask(objs, coco):
            # modify segmentation by reference
            for obj in objs:
                t = coco.imgs[obj['image_id']]
                if type(obj['segmentation']) == list:
                    if type(obj['segmentation'][0]) == dict:
                        print('debug')
                    obj['segmentation'] = mask.frPyObjects(obj['segmentation'], t['height'], t['width'])
                    if len(obj['segmentation']) == 1:
                        obj['segmentation'] = obj['segmentation'][0]
                    else:
                        # an object can have multiple polygon regions
                        # merge them into one RLE mask
                        obj['segmentation'] = mask.merge(obj['segmentation'])
                elif type(obj['segmentation']) == dict and type(obj['segmentation']['counts']) == list:
                    obj['segmentation'] = mask.frPyObjects([obj['segmentation']], t['height'], t['width'])[0]
                elif type(obj['segmentation']) == dict and type(obj['segmentation']['counts']) == str:
                    pass
                else:
                    raise Exception('segmentation format not supported.')

        p = self.params
        if p.use_cats:
            gts = self.coco_gt.load_anns(self.coco_gt.get_ann_ids(image_ids=p.image_ids, cat_ids=p.cat_ids))
            dts = self.coco_dt.load_anns(self.coco_dt.get_ann_ids(image_ids=p.image_ids, cat_ids=p.cat_ids))
        else:
            gts = self.coco_gt.load_anns(self.coco_gt.get_ann_ids(image_ids=p.image_ids))
            dts = self.coco_dt.load_anns(self.coco_dt.get_ann_ids(image_ids=p.image_ids))

        # convert ground truth to mask if iou_type == "segm"
        if p.iou_type == "segm":
            _to_mask(gts, self.coco_gt)
            _to_mask(dts, self.coco_dt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iou_type == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.eval_images = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.eval_images
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...      ')
        params = self.params
        # add backward compatibility if use_segmentation is specified in params
        if params.use_segmentation is not None:
            params.iou_type = "segm" if params.use_segmentation == 1 else "bbox"
            print("use_segmentation (deprecated) is not None. Running %s evaluation" % params.iou_type)
        params.image_ids = list(np.unique(params.image_ids))
        if params.use_cats:
            params.cat_ids = list(np.unique(params.cat_ids))
        params.max_dets = sorted(params.max_dets)
        self.params = params

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = params.cat_ids if params.use_cats else [-1]

        if params.iou_type == "segm" or params.iou_type == "bbox":
            compute_iou = self.compute_iou
        elif params.iou_type == "keypoints":
            compute_iou = self.compute_oks
        self.ious = {
            (image_id, cat_id): compute_iou(image_id, cat_id) for image_id in params.image_ids for cat_id in cat_ids}

        max_det = params.max_dets[-1]
        self.eval_images = [self.evaluate_image(image_id, cat_id, area_range, max_det)
                            for cat_id in cat_ids
                            for area_range in params.area_range
                            for image_id in params.image_ids
                            ]
        self._params_eval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t=%0.2fs).' % (toc - tic))

    def compute_iou(self, image_id, cat_id):
        params = self.params
        if params.use_cats:
            gt = self._gts[image_id, cat_id]
            dt = self._dts[image_id, cat_id]
        else:
            gt = [_ for cId in params.cat_ids for _ in self._gts[image_id, cId]]
            dt = [_ for cId in params.cat_ids for _ in self._dts[image_id, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort(map(lambda x: -x["score"], dt), kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > params.max_dets[-1]:
            dt = dt[0:params.max_dets[-1]]

        if params.iou_type == "segm":
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif params.iou_type == "bbox":
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception("unknown iou_type for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = mask.iou(d, g, iscrowd)
        return ious

    def compute_oks(self, image_id, cat_id):
        params = self.params
        # Dimension here should be Nxm
        gts = self._gts[image_id, cat_id]
        dts = self._dts[image_id, cat_id]
        inds = np.argsort(map(lambda x: -x["score"], dts), kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > params.max_dets[-1]:
            dts = dts[0:params.max_dets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluate_image(self, image_id, cat_id, a_range, max_det):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        params = self.params
        if params.use_cats:
            gt = self._gts[image_id, cat_id]
            dt = self._dts[image_id, cat_id]
        else:
            gt = [_ for cId in params.cat_ids for _ in self._gts[image_id, cId]]
            dt = [_ for cId in params.cat_ids for _ in self._dts[image_id, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < a_range[0] or g['area'] > a_range[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind="mergesort")
        gt = map(lambda i: gt[i], gtind)
        dtind = np.argsort([-d['score'] for d in dt], kind="mergesort")
        dt = map(lambda i: dt[i], dtind[0:max_det])
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[image_id, cat_id][:, gtind] if len(self.ious[image_id, cat_id]) > 0 \
            else self.ious[image_id, cat_id]

        T = len(params.iou_thresholds)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gt_ignore = np.array([g['_ignore'] for g in gt])
        dt_ignore = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(params.iou_thresholds):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dt_ignore[tind, dind] = gt_ignore[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < a_range[0] or d['area'] > a_range[1] for d in dt]).reshape((1, len(dt)))
        dt_ignore = np.logical_or(dt_ignore, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': image_id,
            'category_id': cat_id,
            'a_range': a_range,
            'max_det': max_det,
            'dt_ids': [d['id'] for d in dt],
            'gt_ids': [g['id'] for g in gt],
            'dt_matches': dtm,
            'gt_matches': gtm,
            'dt_scores': [d['score'] for d in dt],
            'gt_ignore': gt_ignore,
            'dt_ignore': dt_ignore,
        }

    def accumulate(self, params=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param params: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...   ')
        tic = time.time()
        if not self.eval_images:
            print('Please run evaluate() first')
        # allows input customized parameters
        if params is None:
            params = self.params
        params.cat_ids = params.cat_ids if params.use_cats == 1 else [-1]
        T = len(params.iou_thresholds)
        R = len(params.recall_thresholds)
        K = len(params.cat_ids) if params.use_cats else 1
        A = len(params.area_range)
        M = len(params.max_dets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))

        # create dictionary for future indexing
        _pe = self._params_eval
        cat_ids = _pe.cat_ids if _pe.use_cats else [-1]
        set_k = set(cat_ids)
        set_a = set(map(tuple, _pe.area_range))
        set_m = set(_pe.max_dets)
        set_i = set(_pe.image_ids)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(params.cat_ids) if k in set_k]
        m_list = [m for n, m in enumerate(params.max_dets) if m in set_m]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), params.area_range)) if a in set_a]
        i_list = [n for n, i in enumerate(params.image_ids) if i in set_i]
        I0 = len(_pe.image_ids)
        A0 = len(_pe.area_range)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, max_det in enumerate(m_list):
                    E = [self.eval_images[Nk + Na + i] for i in i_list]
                    E = filter(None, E)
                    if len(E) == 0:
                        continue
                    dt_scores = np.concatenate([e['dt_scores'][0:max_det] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dt_scores, kind='mergesort')

                    dtm = np.concatenate([e['dt_matches'][:, 0:max_det] for e in E], axis=1)[:, inds]
                    dt_ignore = np.concatenate([e['dt_ignore'][:, 0:max_det] for e in E], axis=1)[:, inds]
                    gt_ignore = np.concatenate([e['gt_ignore'] for e in E])
                    np_ignore = np.count_nonzero(gt_ignore == 0)
                    if np_ignore == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dt_ignore))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ignore))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / np_ignore
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, params.recall_thresholds, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
        self.eval = {
            'params': params,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall': recall,
        }
        toc = time.time()
        print('DONE (t=%0.2fs).' % toc - tic)

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iou_threshold=None, area_range='all', max_dets=100):
            p = self.params
            i_str = ' {:<18} {} @[ IoU={:<9} | area={:>6} | max_dets={:>3} ] = {}'
            title_str = 'Average Precision' if ap == 1 else 'Average Recall'
            type_str = '(AP)' if ap == 1 else '(AR)'
            iou_str = '%0.2f:%0.2f' % (p.iou_thresholds[0], p.iou_thresholds[-1]) if iou_threshold is None \
                else '%0.2f' % iou_threshold
            area_str = area_range
            max_dets_str = '%d' % max_dets

            aind = [i for i, a_range in enumerate(p.area_range_label) if a_range == area_range]
            mind = [i for i, m_det in enumerate(p.max_dets) if m_det == max_dets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iou_threshold is not None:
                    t = np.where(iou_threshold == p.iou_thresholds)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iou_threshold is not None:
                    t = np.where(iou_threshold == p.iou_thresholds)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(i_str.format(title_str, type_str, iou_str, area_str, max_dets_str, '%.3f' % float(mean_s)))
            return mean_s

        def _summarize_detections():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iou_threshold=.5)
            stats[2] = _summarize(1, iou_threshold=.75)
            stats[3] = _summarize(1, area_range='small')
            stats[4] = _summarize(1, area_range='medium')
            stats[5] = _summarize(1, area_range='large')
            stats[6] = _summarize(0, max_dets=1)
            stats[7] = _summarize(0, max_dets=10)
            stats[8] = _summarize(0, max_dets=100)
            stats[9] = _summarize(0, area_range='small')
            stats[10] = _summarize(0, area_range='medium')
            stats[11] = _summarize(0, area_range='large')
            return stats

        def _summarize_keypoints():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, max_dets=20)
            stats[1] = _summarize(1, max_dets=20, iou_threshold=.5)
            stats[2] = _summarize(1, max_dets=20, iou_threshold=.75)
            stats[3] = _summarize(1, max_dets=20, area_range='medium')
            stats[4] = _summarize(1, max_dets=20, area_range='large')
            stats[5] = _summarize(0, max_dets=20)
            stats[6] = _summarize(0, max_dets=20, iou_threshold=.5)
            stats[7] = _summarize(0, max_dets=20, iou_threshold=.75)
            stats[8] = _summarize(0, max_dets=20, area_range='medium')
            stats[9] = _summarize(0, max_dets=20, area_range='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iou_type = self.params.iou_type
        if iou_type == "segm" or iou_type == "bbox":
            summarize = _summarize_detections
        elif iou_type == "keypoints":
            summarize = _summarize_keypoints
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''

    def set_detection_params(self):
        self.image_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iou_thresholds = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recall_thresholds = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.max_dets = [1, 10, 100]
        self.area_range = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.area_range_label = ['all', 'small', 'medium', 'large']
        self.use_cats = 1

    def set_keypoint_params(self):
        self.image_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iou_thresholds = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recall_thresholds = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.max_dets = [20]
        self.area_range = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.area_range_label = ['all', 'medium', 'large']
        self.use_cats = 1

    def __init__(self, iou_type="segm"):
        if iou_type == "segm" or iou_type == "bbox":
            self.set_detection_params()
        elif iou_type == "keypoints":
            self.set_keypoint_params()
        else:
            raise Exception("iou_type not supported")
        self.iou_type = iou_type
        # use_segmentation is deprecated
        self.use_segmentation = None
