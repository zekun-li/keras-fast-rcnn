import _init_paths
import cPickle as pickle
import numpy as np
from utils.cython_nms import nms
from datasets.factory import get_imdb
import datasets

'''
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
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes
'''
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


def main():
    datasets.DATA_DIR = '/nfs/isicvlnas01/users/zekunl/datastore/VOCdevkit/VOC2007'
    with open("../22116_all_boxes.pkl","r") as f:
        all_boxes = pickle.load(f)

    for i in xrange(21):
        for j in xrange(4952):
            if all_boxes[i][j] != []:
                all_boxes[i][j] = all_boxes[i][j].astype(np.float32)

    nms_dets = apply_nms(all_boxes,0.3)
    imdb = get_imdb("voc_2007_test")
    imdb._write_voc_results_file(nms_dets)
    print 'nms done'


if __name__ == "__main__":
    main()
