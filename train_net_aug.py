
#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets
import argparse
import pprint
import numpy as np
import sys
# ------------
import roi_data_layer.roidb as rdl_roidb
from keras_model import fastrcnn 
from keras_model import prepare_data
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model
import time
import pickle
import os

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

# fbratio = foreground / background
def background_filter(roidb, fbratio = 1.0):                                                       
    for ind in xrange(len(roidb)):
        img_rois = roidb[ind]
        target_label = img_rois['bbox_targets'][:,0] # vector, target labels for all rois
        fore_indices = np.where(target_label != 0)[0] # vector
        back_indices = np.where(target_label == 0)[0] # vector
        num_fore = len(fore_indices)
        filtered_num_back = int(num_fore/fbratio)
        np.random.shuffle(back_indices)
        filtered_back_indices = back_indices[0:filtered_num_back] # shuffle indices to take background sample from      # vector                       
        filtered_indices = np.concatenate((fore_indices[:,np.newaxis],filtered_back_indices[:,np.newaxis])) # array
        filtered_indices = filtered_indices[:,0] # vector 
        roidb[ind]['box_normalized'] = img_rois['box_normalized'][filtered_indices]
        roidb[ind]['bbox_targets'] = img_rois['bbox_targets'][filtered_indices]
             
    return roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    #parser.add_argument('--gpu', dest='gpu_id',
    #                    help='GPU device id to use [0]',
    #                    default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--data',dest='data_dir',help = 'data directory',default=None,type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


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
        epoch += 1


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

    # if has pretrained model, load weights from hdf5 file
    if args.pretrained_model is not None:
        fastrcnn.fast.load_weights(args.pretrained_model)

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    print 'Computing bounding-box regression targets...'
    # bbox_means, bbox_stds haven't been used so far
    bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)

    print 'loading images ...'
    prepare_data.add_image_data(roidb)
    print 'Computing normalized roi boxes coordinates ...'
    prepare_data.add_normalized_bbox(roidb)
    print 'filtering roidb...'
    roidb = background_filter(roidb)
    print "roidb has {} images".format(len(roidb))

    # ----------------data augmentation ----------------------------------
    print 'augmenting...'
    roidb = np.repeat(roidb,2)
    manip_channel = np.random.randint(0,3) # pick a random channel to manipulate
    for i in xrange(1,len(roidb),2): # manipulte on the copy
        # increase intensity by 30 on manip_channel
        roidb[i]['image_data'][manip_channel] = np.clip(roidb[i]['image_data'][manip_channel]+30,0,255) # min = 0, max = 255


    trn_data_list = roidb[0:10000]  
    #trn = datagen( trn_data_list, nb_epoch = len(trn_data_list) )   
    # generate data infinitely
    trn = datagen( trn_data_list, nb_epoch = -1, mode = 'training' )   
    
    val_data_list = roidb[10000:]
    val = datagen( val_data_list, nb_epoch = -1, mode = 'validation')

    # define callbacks
    csv_logger = CSVLogger('output/aug_slice_2012train.log')
    # saves the model weights after each epoch if the validation loss decreased
    check_point = ModelCheckpoint(filepath = 'output/weights/aug_slice_model.hdf5', monitor = 'loss',save_best_only = True, save_weights_only = False)
    
    
    print "training ..."
    tic  = time.clock()
    history = fastrcnn.fast.fit_generator(trn,samples_per_epoch = 3000 ,nb_epoch = 5000, validation_data = val, nb_val_samples = len(val_data_list),callbacks = [csv_logger,check_point],max_q_size=2) 
    toc = time.clock()
    print "done training, used %d secs" % (toc-tic)
    
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('output/loss.jpg')
    #plt.show()
    plt.close()

    '''
    i = 0
    X = np.expand_dims(roidb[i]['image_data'],axis = 0)
    R = np.expand_dims(roidb[i]['box_normalized'],axis = 0)
    P = roidb[i]['bbox_targets'][:,0].astype(np.int32) # get label
    P = np.expand_dims(to_categorical(P,21).astype(np.float32),axis = 0)
    B = np.expand_dims(roidb[i]['bbox_targets'][:,1:],axis=0) # get bbox_coordinates
    
    #R = R[:,0:4,:]
    #P = P[:,0:4,:]
    #B = B[:,0:4,:]
    fastrcnn.fast.fit({'batch_of_images': X, 'batch_of_rois': R},{'proba_output':P, 'bbox_output':B}, batch_size = 1, nb_epoch=1,verbose=1)
    '''
