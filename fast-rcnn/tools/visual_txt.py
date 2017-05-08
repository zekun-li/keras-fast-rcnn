#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import time, os, sys
import argparse
import numpy as np
from collections import defaultdict
import cv2
import re

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Visualize txt file')
    parser.add_argument('--imgs',dest='imgs_path',help = 'path of the images',default=None,type=str)
    parser.add_argument('--txts', dest = 'txts_folder', help = 'txt folder path', default = None, type = str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    classes = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
    colors = [(112,25,25),(237, 149,100),(205,90,106),(205,0,0),(255,191,0),(238,238,175),(208,224,64),(170,205,102),(143,188,143),(87,139,46),(113,179,60),(170,178,32),(152,251,152),(47,255,173),(154,250,0),(0,225,225),(32,165,218),(92,92,205),(30,105,210),(114,128,250)]

    imgs_path = args.imgs_path
    txts_folder = args.txts_folder
    dic = defaultdict(list)

    year = re.search('VOC20..',imgs_path).group(0)[-2:]

    for i in xrange(20):
        class_label = classes[i]
        txt_file = txts_folder+'/comp4_det_val_'+class_label+'.txt'
        with open(txt_file) as f:
            for line in f:
                (img_name, conf, xmin, ymin, xmax, ymax) = line.split()
                dic[img_name].append([i,conf, int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])

    for name, bboxes in dic.iteritems():
        img_file = imgs_path + '/' + name + '.jpg'
        img = cv2.imread(img_file)
        for bbox in bboxes:
            (class_index, conf,xmin,ymin,xmax,ymax) = bbox
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax), colors[class_index], thickness = 2)
            cv2.putText(img,classes[class_index] + str(conf),(xmin+2,ymin+15),0,0.5,colors[class_index], 2)
        cv2.imwrite('vis/'+year+'_nms_'+name+'.jpg',img)
