# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:45:35 2016

@author: yue_wu
"""

import sys
import os
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
from keras.layers.core import Dense, Dropout, Activation, Permute, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD,Adadelta
from theano.ifelse import ifelse
from keras.utils.np_utils import to_categorical
from pool import max_pooling,slice_pooling,float_max_pooling # self-defined max pooling

target_h = 7
target_w = 7
nb_classes = 21 # 1000
# coord (1x4 with xmin, ymin, xmax, ymax) has been normalized to (0,1)
# x: 3d tensor
def _per_roi_pooling( coord, x ):
    #x = tt.tensor3() # 512x7x7 float tensor
    #coord = tt.fvector() # [ xmin, ymin, xmax, ymax ] in [0,1] x-width,y-height
    # step 1: float coord to int
    nb_rows = x.shape[1] # height,y
    nb_cols = x.shape[2] # width,x
    icoords = tt.iround(coord * [nb_cols, nb_rows, nb_cols, nb_rows]) # xmin,xmax multiply nb_cols, ymin,ymax multiply nb_rows
    # 0 <= xmin < nb_cols
    xmin = tt.clip(icoords[0],0,nb_cols-1)
    # 0 <= ymin < nb_rows
    ymin = tt.clip(icoords[1],0,nb_rows-1)

    xmax = tt.clip( icoords[2], 1+xmin, nb_cols ) # min(xmax) = 1+xmin, max(xmax) = nb_cols
    ymax = tt.clip( icoords[3], 1+ymin, nb_rows ) # min (ymax) = 1+ymin, max(ymax) = nb_rows
    
    # if xmin == xmax == nb_cols
    xmin = ifelse(tt.eq(xmax,xmin), xmax -1, xmin )
    # if ymin == ymax == nb_rows
    ymin = ifelse(tt.eq(ymax,ymin), ymax -1, ymin )

    # step 2: extract raw sub-stensor
    roi = x[:, ymin:ymax, xmin:xmax ] 
    # step 3: resize raw to target_hx target_w
    '''
    # method1 (slow): upsampling -> downsampling 
    subtensor_h = ymax - ymin
    subtensor_w = xmax - xmin
    # upsample by ( target_h, target_w ) -> ( subtensor_h * target_h, subtensor_w * target_w )
    kernel = tt.ones((target_h, target_w)) # create ones filter
    roi_up,_ =scan(fn=lambda r2d, kernel: kron(r2d,kernel),sequences = roi,non_sequences = kernel)
    # downsample to (target_h, target_w)
    #target = roi_up[:,::subtensor_h,::subtensor_w]
    target = max_pooling(roi_up, subtensor_h, subtensor_w)
    '''
    # method 2
    target = slice_pooling(roi,target_h, target_w)
    #target = float_max_pooling(roi,target_h, target_w)
    return K.flatten( target )

def _per_sample_pooling( x, coords, nb_feat_rows = 7, nb_feat_cols = 7 ):
    # loop through all coord tuple in coords, generate all roi subtensors in one given image
    roi_all,_ = scan(fn = _per_roi_pooling, sequences = coords, non_sequences = [x] ) 
    return roi_all#K.expand_dims( roi_all, 0 )

# x4d: nb_samples x nb_channels x 2D
# coords3d: nb_samples x nb_rois x 4
def roi_pooling( x_coords ) :
    x4d, coords3d = x_coords
    # loop through all samples 
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
vgg = VGG.VGG_19( weights_path = os.path.join( vgg_model_root, 'vgg19_weights.h5' ) )

def create_vgg_featex_classifier() :
    #--------------------------------------------------------------------------
    # create featex 
    #--------------------------------------------------------------------------
    featex = Sequential(name = 'vgg_conv_layers')
    # block 1
    featex.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
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
    # Load initial weights
    #--------------------------------------------------------------------------
    for fl, vl in zip( featex.layers, vgg.layers[:37] ) : # set weights for 0-36 (included) layers
        fl.set_weights( vl.get_weights() )
    
    return featex

def create_vgg_dense():
    dense_model = Sequential(name = 'dense_layers')
    dense_model.add(Dense(4096, input_shape = (512 * target_h * target_w, ), activation = 'relu', name = 'dense1'))
    dense_model.add(Dropout(0.5))
    dense_model.add(Dense(4096, activation = 'relu',name = 'dense2'))
    dense_model.add(Dropout(0.5))

    for dl, vl in zip( dense_model.layers, vgg.layers[38:-1]):
        try:
            dl.set_weights( vl. get_weights())
        except Exception, e:
            print "WARNING: can not set weights for", dl, e, "in bbox_regressor, skipped"

    return dense_model 

def classifier():
    return Dense( nb_classes, activation='softmax', input_shape = (4096,)) 
    
    
def bbox_regressor( nb_classes = nb_classes ) :
     return Dense(4*nb_classes, activation = 'linear', input_shape = (4096,))    
    

def our_categorical_crossentropy(softmax_proba_2d,true_dist_2d, eps = 1e-5):
    return tt.nnet.nnet.categorical_crossentropy(softmax_proba_2d + eps, true_dist_2d)

# softmax_proba: (nb_samples x nb_rois x 21) output softmax probability from fully connected layer
# true_dist: (nb_samples x nb_rois x 21). Labels after one-hot-encoding 
def our_proba_loss(true_dist_3d,softmax_proba_3d):
    loss_tensor,_ = scan( fn = our_categorical_crossentropy, sequences = [softmax_proba_3d, true_dist_3d])
    return loss_tensor

# L1 smooth function
# smooth(x) = 0.5*x^2  if |x|<1
#           = |x|-0.5 otherwise
def smooth_l1(x):
    abs_x = tt.abs_(x)
    return ifelse(tt.lt(abs_x,1),0.5*abs_x*abs_x, abs_x-0.5)

'''
# predicted_box: 1D [xcenter,ycenter,width,height]
# target_box: 1D [label, xcenter,ycenter,width,height]
def one_bbox_loss(target_box,predicted_box):
    x_offset = predicted_box[0] - target_box[1]
    y_offset = predicted_box[1] - target_box[2]
    w_offset = predicted_box[2] - target_box[3]
    h_offset = predicted_box[3] - target_box[4]

    return smooth_l1(x_offset) + smooth_l1(y_offset) + smooth_l1(w_offset) + smooth_l1(h_offset)
'''
# predicted_box : 1D (21*4) [ xcenter0, ycenter0, width0,height0...xcenter21, ycenter21, width21, height21]
# target_box 1D (5) [label,xcenter, ycenter, width, height]
def one_bbox_loss(target_box,predicted_box):
    true_class = target_box[0]
    start = tt.cast(true_class * 4, 'int32')
    x_offset = predicted_box[0+start] - target_box[1]
    y_offset = predicted_box[1+start] - target_box[2]
    w_offset = predicted_box[2+start] - target_box[3]
    h_offset = predicted_box[3+start] - target_box[4]

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
#                dense_layers
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
featex = create_vgg_featex_classifier()
vgg_conv_output = featex( input_x )

pool_roi_output = merge( inputs = [ vgg_conv_output, input_r ], mode = roi_pooling, output_shape = roi_output_shape , name = 'roi_pooling')
dense_output = TimeDistributed(create_vgg_dense(), name = 'dense_output')( pool_roi_output)
#class_pred  = classifier()
#bbox_pred = bbox_regressor()


# define two outputs
proba_output = TimeDistributed( classifier(), name = 'proba_output' )( dense_output )
bbox_output = TimeDistributed( bbox_regressor(), name = 'bbox_output' )( dense_output )
# define model
fast = Model( input = [input_x, input_r], output = [ proba_output, bbox_output ] )
fast.summary()

# TODO: compile model
# 1) you need to compile this model only ONCE
# 2) you should compile this model with your customized loss functions {our_proba_loss}, {our_bbox_loss}

#from keras.metrics import mean_squared_error as our_proba_loss
#from keras.metrics import mean_absolute_error as our_bbox_loss

sgd = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
fast.compile( optimizer = sgd, loss_weights = [ 1., 1. ], 
              loss = { 'proba_output' : our_proba_loss, 'bbox_output' : our_bbox_loss }, metrics = ['accuracy'] )

def datagen( data_list, mode = 'training', nb_epoch = -1 ) :
    epoch = 0
    nb_samples = len( data_list )
    indices = range( nb_samples )
    while ( epoch < nb_epoch ) or ( nb_epoch < 0 ) :
        if ( mode == 'training' ) :
            np.random.shuffle( indices )
        buffer = None
        for idx in indices :
            try :
                X = np.expand_dims( data_list[idx]['image_data'], axis = 0)
                R = np.expand_dims(data_list[idx]['box_normalized'],axis = 0)
                P = data_list[idx]['bbox_targets'][:,0].astype(np.int32) # get label
                P = np.expand_dims(to_categorical(P,21).astype(np.float32),axis = 0)
                B = np.expand_dims(data_list[idx]['bbox_targets'],axis=0) # get label+ bbox_coordinates
                if ( np.any( [ v is None for v in [ X, R, P, B ] ] ) ) :
                    print "Meet None on sample", idx
                else :
                    buffer  = ( { 'batch_of_images' : X ,
                           'batch_of_rois'   : R },
                           { 'proba_output'  : P ,
                           'bbox_output' : B} )
            except Exception, e :
                print "Fail on sample", idx, e
            if ( buffer is not None ) :
                yield buffer 
        epoch += 1
        print "GenEpoch =", epoch


if __name__  == "__main__":
    
    import pickle     
    with open("../../run/roidb.pickle","r") as f:       
        roidb = pickle.load(f)    


    # DEBUG 
    for i in range(0,len(roidb)):
        print i
        X = np.expand_dims(roidb[i]['image_data'],axis = 0)
        R = np.expand_dims(roidb[i]['box_normalized'],axis = 0)
        P = roidb[i]['bbox_targets'][:,0].astype(np.int32) # get label
        P = np.expand_dims(to_categorical(P,21).astype(np.float32),axis = 0)
        B = np.expand_dims(roidb[i]['bbox_targets'],axis=0) # get bbox_coordinates

        #R = R[:,-1:4,:]
        #P = P[:,0:4,:]
        #B = B[:,0:4,:]
        fast.predict({'batch_of_images': X, 'batch_of_rois': R})
        #fastrcnn.fast.fit({'batch_of_images': X, 'batch_of_rois': R},{'proba_output':P, 'bbox_output':B}, batch_size = 1, nb_epoch=1,verbose=1)
    

        
    # USE of GENERATOR
    '''
    trn_data_list = roidb[0:8]  
    trn = datagen( trn_data_list, nb_epoch = len(trn_data_list) )   
   
    for X_lut, Y_lut in trn :
        print X_lut, Y_lut                                                                            
   
    
    #val_data_list = roidb[8:]                                      
    #val = datagen( val_data_list, nb_epoch = len(val_data_list), mode = 'validation' )  
    
    # Note: nb_epoch should be <= len(sequence of both trn or val)
    fast.fit_generator(trn,samples_per_epoch = 1 ,nb_epoch = 10) 
    #fast.fit_generator(trn,samples_per_epoch = 1 ,nb_epoch = 10, validation_data = val, nb_val_samples = len(val_dat
    '''
    
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
    print fast.predict({'batch_of_images': X, 'batch_of_rois': R})
    print fast.evaluate({'batch_of_images':X, 'batch_of_rois':R},{'proba_output':P, 'bbox_output':B},batch_size = 1,verbose=1)
    fast.fit({'batch_of_images': X, 'batch_of_rois': R},{'proba_output':P, 'bbox_output':B}, batch_size = 1, nb_epoch=1,verbose=1)
    '''
