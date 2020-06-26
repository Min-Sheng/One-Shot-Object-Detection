# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
from collections import OrderedDict

class fss_cell(imdb):
  def __init__(self, image_set, year):
    imdb.__init__(self, 'fss_cell_' + year + '_' + image_set)
    # COCO specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    # name, paths
    self._year = year
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'fss_cell')

    # load COCO API, classes, class <-> id mappings
    self._COCO = COCO(self._get_ann_file())
    cats = self._COCO.loadCats(self._COCO.getCatIds())
    # class name
    self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    # class name to ind (0~14) 0= __background__
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    # class name to cat_id (1~14)
    self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                               self._COCO.getCatIds())))

    self._image_index = self._load_image_set_index()

    
    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)
    
    # Dataset splits that have ground-truth annotations
    self._gt_splits = ('train', 'test')

    self.cat_data = {}

    for i in self._class_to_ind.values():
      # i = 1~14
      self.cat_data[i] = []

  def _get_ann_file(self):
    prefix = 'instances'
    return osp.join(self._data_path,
                    prefix + '_fss_cell_poly_' + self._image_set + '.json')
    
  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._COCO.getImgIds()
    image_ids.sort()
    return image_ids

  def _get_widths(self):
    anns = self._COCO.loadImgs(self._image_index)
    widths = [ann['width'] for ann in anns]
    return widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    file_name = [image_info for image_info in self._COCO.dataset['images'] if image_info['id'] == index][0]['file_name']
    image_path = osp.join(self._data_path, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        [roidb, self.cat_data] = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb
    

    gt_roidb = [self._load_coco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump([gt_roidb,self.cat_data], fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_coco_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations.
    """
    im_ann = self._COCO.loadImgs(index)[0]
    im_path = self.image_path_from_index(index)
    width = im_ann['width']
    height = im_ann['height']

    annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
    objs = self._COCO.loadAnns(annIds)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for i, obj in enumerate(objs):
      x1 = np.max((0, obj['bbox'][0]))
      y1 = np.max((0, obj['bbox'][1]))
      x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
      
        entry = {
              'boxes': obj['clean_bbox'],
              'image_path': im_path,
              'area': obj['area'],
              }
        
        self.cat_data[obj['category_id']].append(entry)
      
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    

    for ix, obj in enumerate(objs):
      cls = obj['category_id']
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()

    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def _do_detection_eval(self, res_file, output_dir, post_fix=None, save=False):
    ann_type = 'bbox'

    tmp = [i-1 for i in self.list]

    coco_dt = self._COCO.loadRes(res_file)

    cocoEval = customCOCOeval(self._COCO, coco_dt, "bbox")
    cocoEval.params.imgIds = self._image_index
    cocoEval.evaluate()
    # print(cocoEval.ious)
    cocoEval.accumulate()
    cocoEval.summarize(class_index=tmp)

    if post_fix:
      eval_file = osp.join(output_dir, ('cocoeval_' + 
                                        post_fix + 
                                        '_results.pkl'))
    else:
      eval_file = osp.join(output_dir, 'detection_results.pkl')
    if save:
      with open(eval_file, 'wb') as fid:
        pickle.dump(cocoEval, fid, pickle.HIGHEST_PROTOCOL)
      print('Wrote COCO eval results to: {}'.format(eval_file))
    return cocoEval

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind]
      if dets == []:
        continue
      dets = np.array(dets).astype(np.float)
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      for k in range(len(dets)):
        results.extend(
          [{'image_id': index,
            'category_id': cat_id,
            'bbox': [xs[k], ys[k], ws[k], hs[k]],
            'score': scores[k]} ])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def evaluate_detections(self, all_boxes, output_dir, post_fix=None, save=False):

    if post_fix:
      res_file = osp.join(output_dir, ('detections_' +
                                        post_fix + 
                                        '_results'))
    else:
      res_file = osp.join(output_dir, ('detections_' +
                                      self._image_set +
                                      self._year +
                                      '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    #if self._image_set.find('test') == -1:
    coco_eval = self._do_detection_eval(res_file, output_dir, post_fix, save)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)
    box_results = _coco_eval_to_box_results(coco_eval)
    self.log_copy_paste_friendly_results(box_results)
    return box_results

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def filter(self, seen=1):

    # if want to use train_categories, seen = 1 
    # if want to use test_categories , seen = 2
    # if want to use both            , seen = 3

    folds = {
    'all': set(range(1, 15)),
    1: set(range(1, 15)) - set(range(1, 3)),
    2: set(range(1, 15)) - set(range(3, 6)),
    3: set(range(1, 15)) - set(range(6, 9)),
    4: set(range(1, 15)) - set(range(9, 11)),
    5: set(range(1, 15)) - set(range(11, 15)),
    }

    if seen==1:
      self.list = cfg.train_categories
      # Group number to class
      if len(self.list)==1:
        self.list = list(folds[self.list[0]])

    elif seen==2:
      self.list = cfg.test_categories
      # Group number to class
      if len(self.list)==1:
          self.list = list(folds['all'] - folds[self.list[0]])

    elif seen==3:
      self.list = cfg.train_categories + cfg.test_categories
      # Group number to class
      if len(self.list)==0:
        self.list =list(folds['all'])

    self.inverse_list = self.list
    # Which index need to be remove
    all_index = list(range(len(self._image_index)))

    for index, info in enumerate(self.roidb):
      for cat in info['gt_classes']:
        if cat in self.list:
            all_index.remove(index)
            break

    # Remove index from the end to start
    all_index.reverse()
    for index in all_index:
      self._image_index.pop(index)
      self.roidb.pop(index)
  
  def log_copy_paste_friendly_results(self, results):
    """Log results in a format that makes it easy to copy-and-paste in a
    spreadsheet. Lines are prefixed with 'copypaste: ' to make grepping easy.
    """
    for task, metrics in results.items():
        print('copypaste: Task: {}'.format(task))
        metric_names = metrics.keys()
        metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
        print('copypaste: ' + ','.join(metric_names))
        print('copypaste: ' + ','.join(metric_vals))


class customCOCOeval(COCOeval):
    
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        super().__init__(cocoGt, cocoDt, iouType)
        self.params.maxDets = [1, 10, 100, 200, 500]
        self.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 64 ** 2], [64 ** 2, 1e5 ** 2]]
        self.params.areaRngLbl = ['all', 'small', 'medium', 'large']

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.useSegm:
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        else:
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        ious = COCOmask.iou(d,g,[])
        return ious
    
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        #
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        
        for g in gt:
          #if 'ignore' not in g:
              #g['ignore'] = 0
          #if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
          #    g['_ignore'] = 1
          if (g['area']<aRng[0] or g['area']>aRng[1]):
              g['_ignore'] = 1
          else:
              g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gt = sorted(gt, key=lambda x: x['_ignore'])
        gtind = [ind for (ind, g) in sorted(enumerate(gt), key=lambda ind_g: ind_g[1]['_ignore']) ]

        gt = [gt[ind] for ind in gtind]
        dt = sorted(dt, key=lambda x: -x['score'])[0:maxDet]
        # load computed ious
        N_iou = len(self.ious[imgId, catId])
        ious = self.ious[imgId, catId][0:maxDet, np.array(gtind)] if N_iou >0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, continue
                        if gtm[tind,gind]>0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
    
    def summarize(self, class_index=None, verbose=1):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=500 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,:,class_index,aind,mind]
                else:
                    s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,class_index,aind,mind]
                else:
                    s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if verbose > 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((14,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[4])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[4])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[4])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[4])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[4])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[3])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[4])
            stats[11] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[4])
            stats[12] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[4])
            stats[13] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[4])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self, cass_index=None):
        self.summarize(class_index)

# Indices in the stats array for COCO boxes and masks
COCO_AP = 0
COCO_AP50 = 1
COCO_AP75 = 2
COCO_APS = 3
COCO_APM = 4
COCO_APL = 5

# ---------------------------------------------------------------------------- #
# Helper functions for producing properly formatted results.
# ---------------------------------------------------------------------------- #

def _coco_eval_to_box_results(coco_eval):
    res = _empty_box_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['box']['AP'] = s[COCO_AP]
        res['box']['AP50'] = s[COCO_AP50]
        res['box']['AP75'] = s[COCO_AP75]
        res['box']['APs'] = s[COCO_APS]
        res['box']['APm'] = s[COCO_APM]
        res['box']['APl'] = s[COCO_APL]
    return res


def _empty_box_results():
    return OrderedDict({
        'box':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APs', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })