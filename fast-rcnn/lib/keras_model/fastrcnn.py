# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:45:35 2016

@author: yue_wu
"""

import os
#gpu_id = '1'
gpu_id = os.environ["SGE_GPU"]
print gpu_id
os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32" % gpu_id
print os.environ["THEANO_FLAGS"]
import sys
# sys.path.insert(0, '/nfs/isicvlnas01/users/yue_wu/thirdparty/keras_1.1.2/keras/' )
import keras
print keras.__version__

from keras.layers import Input, Dense, Merge, Lambda, merge
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from theano import scan
from theano import tensor as tt
from theano import function
from theano.tensor.slinalg import kron
from theano.tensor.signal import pool
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.models import Model,Sequential
from keras.engine.topology import Layer
from keras.layers.core import Dense, Dropout, Activation, Permute, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD
from theano.ifelse import ifelse


target_h = 7
target_w = 7
nb_classes = 21 # 1000
def _per_roi_pooling( coord, x ):
    #target_h = 4
    #target_w = 4
    #x = tt.tensor3() # 512x7x7 float tensor
    #coord = tt.fvector() # [ xmin, ymin, xmax, ymax ] in [0,1] x-width,y-height
    # step 1: float coord to int
    nb_rows = x.shape[1] # height,y
    nb_cols = x.shape[2] # width,x
    icoords = tt.iround(coord * nb_rows)
    #icoords[0] = tt.iround( coord[0] * nb_cols) # assume coord has been normalized to (0,1)
    #icoords[1] = tt.iround( coord[1] * nb_rows)
    #icoords[2] = tt.iround( coord[2] * nb_cols)
    #icoords[3] = tt.iround( coord[3] * nb_rows)
    #xmin = icoords[0]
    #ymin = icoords[1]
    # 0 <= xmin <= nb_cols
    xmin = tt.clip(icoords[0],0,nb_cols)
    # 0 <= ymin <= nb_rows
    ymin = tt.clip(icoords[1],0,nb_rows)

    xmax = tt.clip( icoords[2], 1+xmin, nb_cols ) # min(xmax) = 1+xmin, max(xmax) = nb_cols
    ymax = tt.clip( icoords[3], 1+ymin, nb_rows ) # min (ymax) = 1+ymin, max(ymax) = nb_rows
    
    # if xmin == xmax == nb_cols
    xmin = ifelse(tt.eq(xmax,xmin), xmax -1, xmin )
    # if ymin == ymax == nb_rows
    ymin = ifelse(tt.eq(ymax,ymin), ymax -1, ymin )

    # step 2: extract raw sub-stensor
    roi = x[:, ymin:ymax, xmin:xmax ] 
    # step 3: resize raw to 4x4
    subtensor_h = ymax - ymin
    subtensor_w = xmax - xmin
    # upsample by ( target_h, target_w ) -> ( subtensor_h * target_h, subtensor_w * target_w )
    kernel = tt.ones((target_h, target_w)) # create ones filter
    roi_up,_ =scan(fn=lambda r2d, kernel: kron(r2d,kernel),sequences = roi,non_sequences = kernel)
    # downsample to (target_h, target_w)
    target = roi_up[:,::subtensor_h,::subtensor_w]
    return K.flatten( target )

def _per_sample_pooling( x, coords, nb_feat_rows = 7, nb_feat_cols = 7 ):
    # loop through all coord tuple in coords, generate all roi subtensors in one given image
    roi_all,_ = scan(fn = _per_roi_pooling, sequences = coords, non_sequences = [x] ) 
    return roi_all#K.expand_dims( roi_all, 0 )

def roi_pooling( x_coords ) :
    x4d, coords3d = x_coords
    batch_roi_all, _ = scan( fn = _per_sample_pooling, sequences = [ x4d, coords3d ] )
    return batch_roi_all

def roi_output_shape( input_shapes ):
    x_shape, roi_shape = input_shapes
    nb_samples, nb_chs, _, _ = x_shape
    _, nb_rois, _ = roi_shape
    output_shape = ( nb_samples, nb_rois, nb_chs * target_h * target_w )
    return tuple( output_shape )

vgg_model_root = '/nfs/isicvlnas01/users/yue_wu/thirdparty/keras-model-zoo/models/VGG-19/'
sys.path.insert( 0, vgg_model_root )
import model as VGG
def create_vgg_featex_classifier() :
    vgg = VGG.VGG_19( weights_path = os.path.join( vgg_model_root, 'vgg19_weights.h5' ) )
    #--------------------------------------------------------------------------
    # create featex 
    #--------------------------------------------------------------------------
    featex = Sequential()
    # block 1
    featex.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    featex.add(Convolution2D(64, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(64, 3, 3, activation='relu'))
    featex.add(MaxPooling2D((2,2), strides=(2,2)))
    # block 2
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(128, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(128, 3, 3, activation='relu'))
    featex.add(MaxPooling2D((2,2), strides=(2,2)))
    # block 3
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(256, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(256, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(256, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(256, 3, 3, activation='relu'))
    featex.add(MaxPooling2D((2,2), strides=(2,2)))
    # block 4
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(MaxPooling2D((2,2), strides=(2,2)))
    # block 5
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(ZeroPadding2D((1,1)))
    featex.add(Convolution2D(512, 3, 3, activation='relu'))
    featex.add(MaxPooling2D((2,2), strides=(2,2)))
    #--------------------------------------------------------------------------
    # create classifier
    #--------------------------------------------------------------------------
    classifier = Sequential()
    classifier.add(Dense(4096, activation='relu', input_shape = ( 512*7*7, ) ) )
    classifier.add(Dropout(0.5))
    classifier.add(Dense(4096, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense( nb_classes, activation='softmax') )
    #--------------------------------------------------------------------------
    # Load initial weights
    #--------------------------------------------------------------------------
    for fl, vl in zip( featex.layers, vgg.layers[:37] ) :
        fl.set_weights( vl.get_weights() )
    for cl, vl in zip( classifier.layers, vgg.layers[38:] ) :
        try :
            cl.set_weights( vl.get_weights() )
        except Exception, e :
            print "WARNING: cannot set weights for", cl, e, "skipped"
    return featex, classifier
    
def bbox_regressor( nb_classes = nb_classes ) :
    '''this model expects an input of shape {nb_roi}x{512*7*7} and predicts a bbox of shape {nb_roi}x4
    '''
    bbox = Sequential()
    bbox.add(Dense(1024, input_shape = (512 * target_h * target_w, ), activation = 'relu'))
    bbox.add(Dropout(0.5))
    bbox.add(Dense(256, activation = 'relu'))
    bbox.add(Dropout(0.5))
    bbox.add(Dense(4, activation = 'linear'))
    return bbox
    '''
    bbox = Sequential()
    # bbox.add(Dense( 1024, input_shape = (512 * target_h * target_w, ), activation = 'relu' ) )
    Reshape(512,target_h, target_w)
    bbox.add(Convolution2D(1024,target_h, target_w,border_mode = 'valid',activation = 'relu') )
    bbox.add(Dropout(0.5))
    #bbox.add(Dense( 256, activation='relu') )
    bbox.add(Convolution2D(256, target_h, target_w, border_mode = 'valid',activation = 'relu'))
    bbox.add(Dropout(0.5))
    #bbox.add(Dense( nb_classes * 4, activation='linear') )
    bbox.add(Convolution2D( 64, target_h, target_w, border_mode = 'valid', activation = 'linear'))
    return bbox 
    '''


def our_categorical_crossentropy(softmax_proba_2d,true_dist_2d, eps = 1e-5):
    return tt.nnet.nnet.categorical_crossentropy(softmax_proba_2d + eps, true_dist_2d)

# wrong - softmax_proba: 2D (nb_rois*21), output softmax probability from fully connected layer
# wrong - true_dist: 1D (nb_rois) vector of ints, each int is the class_label
# corrected: softmax_proba and true_dist are both 3D, needs one-hot-encoding before passing params 
def our_proba_loss(true_dist_3d,softmax_proba_3d):
    loss_tensor,_ = scan( fn = our_categorical_crossentropy, sequences = [softmax_proba_3d, true_dist_3d])
    return loss_tensor
    #return tt.nnet.nnet.categorical_crossentropy(softmax_proba + eps, true_dist)

# L1 smooth function
# smooth(x) = 0.5*x^2  if |x|<1
#           = |x|-0.5 otherwise
def smooth_l1(x):
    abs_x = tt.abs_(x)
    return ifelse(tt.lt(abs_x,1),0.5*abs_x*abs_x, abs_x-0.5)

# predicted_box: 1D [xcenter,ycenter,width,height]
# target_box: 1D [label, xcenter,ycenter,width,height]
def one_bbox_loss(target_box,predicted_box):
    x_offset = predicted_box[0] - target_box[1]
    y_offset = predicted_box[1] - target_box[2]
    w_offset = predicted_box[2] - target_box[3]
    h_offset = predicted_box[3] - target_box[4]

    return smooth_l1(x_offset) + smooth_l1(y_offset) + smooth_l1(w_offset) + smooth_l1(h_offset)

# predicted_box: 1D [xcenter,ycenter,width,height]
# target_box: 1D [label, xcenter,ycenter,width,height]
# if target label is 0 (background), then set the loss to be zero, instead of calculating smooth_l1
def one_bbox_loss_filter(target_box, predicted_box):
    return ifelse(tt.eq(target_box[0],0), 0.0, one_bbox_loss(target_box,predicted_box))

def nb_rois_bbox_loss(target_box_2d, predicted_box_2d):
    loss_2d , _ = scan(fn = one_bbox_loss_filter,sequences = [target_box_2d,predicted_box_2d])
    return loss_2d

def our_bbox_loss ( target_box_3d, predicted_box_3d):
    loss, _ = scan(fn = nb_rois_bbox_loss,sequences = [target_box_3d,predicted_box_3d])
    return loss


#################################################################################
# Main
#################################################################################
"""
We need to construct a (graph) model like below
#
# INPUT: batch_of_images   #INPUT: batch_of_rois
# ----------------------   ----------------------           
#          |                        |
#     [ featex ]                    |
#          |                        |
#   vgg_conv_tensor                 |  
# ---------------------             |
#          |                        |
#         [ our_____roi_____pooling  ] 
#                     |
#             pooled_roi_tensor   
#                     |
#        ------------------------------
#          |                        |      
# [ roi_cls_predictor ]   [ roi_bbox_predictor ]
#          |                        |
# #OUTPUT: batch_of_cls    #OUTPUT: batch_of_bbox 
#
NOTE: 
1) the loaded pretrained VGG19 model expects input image to be normalized
   below is an example of predicting object class for a single image
       
       im = cv2.resize(cv2.imread( imfile ), (224, 224)).astype(np.float32)
       im[:,:,0] -= 103.939
       im[:,:,1] -= 116.779
       im[:,:,2] -= 123.68
       im = im.transpose((2,0,1))
       im = np.expand_dims(im, axis=0)
       # Test pretrained model                                                                              
       out = vgg19.predict(im).ravel()
2) you may rewrite the [ roi_bbox_predictor ] module
3) make sure provided target values in training compatible with model outputs
"""
from keras.layers.wrappers import TimeDistributed
# nb_rois = 4
# define two inputs
# input_x = Input( shape = ( 3, 224, 224 ), name = 'batch_of_images' )
# input_r = Input( shape = ( nb_rois, 4 ), name = 'batch_of_rois' )

input_x = Input( shape = ( 3, None, None ), name = 'batch_of_images' )
input_r = Input( shape = ( None, 4 ), name = 'batch_of_rois' )  
# define four major modules
featex, classifier = create_vgg_featex_classifier()
vgg_conv_output = featex( input_x )
pool_roi_output = merge( inputs = [ vgg_conv_output, input_r ], mode = roi_pooling, output_shape = roi_output_shape )
bbox_pred = bbox_regressor()
# define two outputs
proba_output = TimeDistributed( classifier, name = 'proba_output' )( pool_roi_output )
bbox_output = TimeDistributed( bbox_pred, name = 'bbox_output' )( pool_roi_output )
# define model
fast = Model( input = [input_x, input_r], output = [ proba_output, bbox_output ] )
fast.summary()
# TODO: compile model
# 1) you need to compile this model only ONCE
# 2) you should compile this model with your customized loss functions {our_proba_loss}, {our_bbox_loss}

#from keras.metrics import mean_squared_error as our_proba_loss
#from keras.metrics import mean_absolute_error as our_bbox_loss

fast.compile( optimizer = 'sgd', loss_weights = [ 1., 1. ], 
              loss = { 'proba_output' : our_proba_loss, 'bbox_output' : our_bbox_loss } )

if __name__  == "__main__":
    '''
    # test code
    x = np.random.randn( 2, 3, 224, 224 ).astype( np.float32 )
    roi = np.array( [ [0,0,1,1], [0,0,.75,.75], [0,0,.5,.5], [0,0,.25,.25]] ).astype( np.float32 ).reshape( [1,-1,4] )
    roi3d = np.concatenate( [ roi, roi ] )
    y = fast.predict( {'batch_of_images' : x, 'batch_of_rois' : roi3d } )
    print 'proba_output.shape =', y[0].shape
    print 'bbox_output.shape =', y[1].shape
    '''

    '''
    # if case we can all training data in memory, then you can train the model as follows
    from keras.utils.np_utils import to_categorical

    nb_samples = 1
    nb_rois = 40
    X = np.random.randint(0, 225, ( nb_samples, 3, 224, 224 )).astype( np.float32 )
    R = np.concatenate( [ np.zeros( (nb_samples, nb_rois, 2 ) ), np.ones(( nb_samples, nb_rois, 2 ) ) ], axis = -1 ).astype( np.float32 )
    P = to_categorical( np.random.choice( range( nb_classes ), ( nb_samples, nb_rois ) ).ravel(), nb_classes +1 ).reshape( ( nb_samples, nb_rois, -1 ) )
    #B = np.random.rand( nb_samples, nb_rois, nb_classes * 4 )
    B = np.random.randn(nb_samples,nb_rois,4)
    fast.fit( { 'batch_of_images' : X, 'batch_of_rois' : R }, {'proba_output' : P, 'bbox_output' : B }, batch_size = 1, nb_epoch = 1, verbose = 1 )

    '''
    import pickle
    from keras.utils.np_utils import to_categorical
    
    with open("../../run/roidb.pickle","r") as f:
        roidb = pickle.load(f)
    
    i = 3
    X = np.expand_dims(roidb[i]['image_data'],axis = 0)
    R = np.expand_dims(roidb[i]['box_normalized'],axis = 0)
    P = roidb[i]['bbox_targets'][:,0].astype(np.int32) # get label
    P = np.expand_dims(to_categorical(P,21).astype(np.float32),axis = 0)
    #B = np.expand_dims(roidb[i]['bbox_targets'][:,1:],axis=0) # get bbox_coordinates
    B = np.expand_dims(roidb[i]['bbox_targets'],axis=0) # get label + bbox_coordinates
    
    print R
    print P
    print B
    
    print fast.predict({'batch_of_images': X, 'batch_of_rois': R})
    print fast.evaluate({'batch_of_images':X, 'batch_of_rois':R},{'proba_output':P, 'bbox_output':B},batch_size = 1,verbose=1)
    fast.fit({'batch_of_images': X, 'batch_of_rois': R},{'proba_output':P, 'bbox_output':B}, batch_size = 1, nb_epoch=1,verbose=1)
