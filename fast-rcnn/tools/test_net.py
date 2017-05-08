#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import datasets
import argparse
import pprint
import time, os, sys

# ------------
from keras_model import fastrcnn
from keras_model import prepare_data
from keras.utils.np_utils import to_categorical
import numpy as np
import cPickle as pickle
import os
import time
import roi_data_layer.roidb as rdl_roidb
from utils.cython_nms import nms

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1e-14
    heights = boxes[:, 3] - boxes[:, 1] + 1e-14
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    _, height, width = im_shape
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[0]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], width - 1)
    # y2 < im_shape[1]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], height - 1)
    return boxes

# actually only one box
def compute_iou(query_boxes, boxes):
    overlap = 0
    box_area = (
            (query_boxes[2] - query_boxes[0] + 1) *
            (query_boxes[3] - query_boxes[1] + 1)
    )
    iw = (
            min(boxes[2], query_boxes[2]) -
            max(boxes[0], query_boxes[0]) + 1
        )
    if iw > 0:
        ih = (
            min(boxes[3], query_boxes[3]) -
            max(boxes[1], query_boxes[1]) + 1
        )
        if ih > 0:
            ua = float(
                (boxes[2] - boxes[0] + 1) *
                (boxes[3] - boxes[1] + 1) +
                box_area - iw * ih
            )
            overlap = iw * ih / ua
    return overlap
    
def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_out_gt_proposals(roidb):
    for ind in xrange(len(roidb)):
        img_rois = roidb[ind]
        use_indices = np.where(img_rois['gt_classes'] == 0) # indices of none-gt classes
        roidb[ind]['boxes'] = img_rois['boxes'][use_indices]
        roidb[ind]['box_normalized'] = img_rois['box_normalized'][use_indices]
        roidb[ind]['bbox_targets'] = img_rois['bbox_targets'][use_indices]

    return roidb

def datagen( data_list, mode = 'training', nb_epoch = -1 ) :
    epoch = 0
    nb_samples = len( data_list )
    indices = range( nb_samples )
    while ( epoch < nb_epoch ) or ( nb_epoch < 0 ) :
        if ( mode == 'training' ) :
            np.random.shuffle( indices )
        for idx in indices :
            X =  np.expand_dims( data_list[idx]['image_data'], axis = 0)
            R = np.expand_dims(data_list[idx]['box_normalized'],axis = 0)
            P = data_list[idx]['bbox_targets'][:,0].astype(np.int32) # get label
            P = np.expand_dims(to_categorical(P,21).astype(np.float32),axis = 0)
            B = np.expand_dims(data_list[idx]['bbox_targets'],axis=0) # get label+ bbox_coordinates
   
            yield ( { 'batch_of_images' : X ,
                      'batch_of_rois'   : R },
                    { 'proba_output'  : P ,
                      'bbox_output' : B} )
            print 'yield with idx= '+str(idx)
        epoch += 1
        print epoch

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--data',dest='data_dir',help = 'data directory',default=None,type=str)
    parser.add_argument('--weight', dest = 'weight_file', help = 'hdf5 weight file', default = None, type = str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    
    if args.data_dir is not None:
        datasets.DATA_DIR = args.data_dir

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    print 'loading images ...'
    prepare_data.add_image_data(roidb)
    print 'Computing normalized roi boxes coordinates ...'
    prepare_data.add_normalized_bbox(roidb)
    print "roidb has {} images".format(len(roidb))
    print 'done'


    assert os.path.exists(args.weight_file) 
    # load weights saved from trained model
    fastrcnn.fast.load_weights(args.weight_file)

    # ---------------------un-norm bbox weights ----------------
    norm_w = fastrcnn.fast.get_layer('bbox_output').get_weights()[0]
    norm_b = fastrcnn.fast.get_layer('bbox_output').get_weights()[1]

    with open('unnorm.params.2012train','r') as f:
        params = pickle.load(f)

    fastrcnn.fast.get_layer('bbox_output').set_weights((norm_w * (params['bbox_stds'][np.newaxis,:]), norm_b* params['bbox_stds']+ params['bbox_means']))

    # ---------------------------------------------------------------
    
    #print 'Computing bounding-box regression targets...'
    # bbox_means, bbox_stds haven't been used so far
    #bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)

    #roidb = filter_out_gt_proposals(roidb)
    # all_boxes[cls][image]
    all_boxes = [[[] for _ in xrange(len(roidb))] for _ in xrange(20+1)]
    for i in range(0,len(roidb)):
        if i%100 == 0:
            print i
        X = np.expand_dims(roidb[i]['image_data'],axis = 0)
        R = np.expand_dims(roidb[i]['box_normalized'],axis = 0)
        # start prediction
        proba_out, bbox_delta_out = fastrcnn.fast.predict({'batch_of_images': X, 'batch_of_rois': R})
        #print proba_out.shape, bbox_delta_out.shape
        orig_boxes = roidb[i]['boxes']
        bbox_delta_out = bbox_delta_out[0]
        pred_boxes = _bbox_pred(orig_boxes, bbox_delta_out) # this includes gt boxes on the top
        pred_labels = np.argmax(proba_out[0],axis = 1)
        pred_conf = np.max(proba_out[0], axis = 1)

        im_shape = X[0].shape
        pred_boxes = _clip_boxes(pred_boxes, im_shape)
        for roi_ind in range(len(pred_boxes)):
            # skip ground truth bboxes.
            if roidb[i]['gt_classes'][roi_ind] != 0:
                continue

            label = pred_labels[roi_ind]
            conf = pred_conf[roi_ind]
            box = pred_boxes[roi_ind][label*4:label*4+4]
            box = box.astype(np.float32)
            if all_boxes[label][i] == []:
                all_boxes[label][i] = np.expand_dims(np.append(conf, box),axis =0)
            else:
                new_line = np.expand_dims( np.append(conf,box), axis =0)
                all_boxes[label][i] = np.append( all_boxes[label][i], new_line, axis =0)
    
    '''
    # --------save all_boxes ---------------
    pid = os.getpid()
    filename = str(pid)+'_all_boxes.pkl'
    with open(filename,'w') as f:
        pickle.dump(all_boxes, f)
    '''
    print 'applying nms...'
    # apply nms
    nms_dets = apply_nms(all_boxes, 0.3)    
    imdb._write_voc_results_file(nms_dets)
    print 'prediction done'
