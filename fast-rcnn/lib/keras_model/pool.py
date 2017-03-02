import os
gpu_id = os.environ["SGE_GPU"]
#gpu_id = '0'
print gpu_id
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32,profile=False" % gpu_id
print os.environ["THEANO_FLAGS"]

import theano.tensor as tt
from theano import function,scan
import numpy as np

# order of papameters need to follow theano scan function convention: [sequences,non_sequences]
def compute_max(start_x, roi_up, bin_size_x, start_y, bin_size_y):
    end_x = start_x+bin_size_x 
    end_y = start_y+bin_size_y 
    max_along_channel= roi_up[:,start_y:end_y,start_x:end_x].max(axis =-1).max(axis = -1) # should compute tensor max
    return max_along_channel
    #return start_x

def for_x(start_y, roi_up,bin_size_x,bin_size_y):
    _, size_y,size_x = roi_up.shape
    start_indices_x = tt.arange(size_x)[::bin_size_x]
    max_along_x,_ = scan(fn = compute_max, non_sequences=[roi_up, bin_size_x, start_y, bin_size_y], sequences = [start_indices_x]) 
    return max_along_x

# size_y should be divisible by bin_size_y and size_x should be divisible by bin_size_x 
def max_pooling( roi_up, bin_size_y, bin_size_x):
    nb_channels, size_y, size_x = roi_up.shape
    start_indices_y = tt.arange(size_y)[::bin_size_y]
    pool_out, _= scan(fn = for_x, sequences = [start_indices_y], non_sequences = [roi_up, bin_size_x, bin_size_y])
    return pool_out.dimshuffle(2,0,1)


# WxH -> axH -> axb
def slice_pooling( roi, target_y, target_x):
    nb_channels, size_y, size_x = roi.shape
    # WxH -> axH
    scale_x = 1.0* size_x /target_x # scale to shrink
    slice_indices_x =  tt.iround( tt.arange(target_x)*scale_x) # indices from which to take slices
    slice_indices_x = tt.clip(slice_indices_x, 0, size_x -1) # min = 0, max= size_x-1
    # slice along x axis
    roi = roi[:,:,slice_indices_x]
    # axH -> axb
    scale_y = 1.0*size_y/target_y
    slice_indices_y = tt.iround( tt.arange(target_y)*scale_y)
    slice_indices_y = tt.clip(slice_indices_y, 0, size_y-1) # maximum can not be greater than size_y-1
    roi = roi[:,slice_indices_y,:]
    return roi


# WxH -> Wxb -> axb
def slice_pooling1( roi, target_y, target_x):
    nb_channels, size_y, size_x = roi.shape
    # WxH -> Wxb
    scale_y = 1.0*size_y/target_y
    slice_indices_y = tt.iround( tt.arange(target_y)*scale_y)
    slice_indices_y = tt.clip(slice_indices_y, 0, size_y-1)
    roi = roi[:,slice_indices_y,:]
    # Wxb -> axb
    scale_x = 1.0* size_x /target_x # scale to shrink
    slice_indices_x =  tt.iround( tt.arange(target_x)*scale_x) # indices from which to take slices
    slice_indices_x = tt.clip(slice_indices_x, 0, size_x -1) # min = 0, max= size_x-1
    # slice along x axis
    roi = roi[:,:,slice_indices_x]
    
    return roi

if __name__=="__main__":

    bin_size_x = tt.iscalar() # subtensor_w
    bin_size_y = tt.iscalar() # subtensor_h
    #start_x = tt.iscalar()
    #start_y = tt.iscalar()
    roi_up = tt.tensor3()
    _, size_y, size_x = roi_up.shape


    #out = max_pooling( roi_up, bin_size_y, bin_size_x)
    out = slice_pooling( roi_up, bin_size_y, bin_size_x)
    out1 = slice_pooling1( roi_up, bin_size_y, bin_size_x)
    func = function([roi_up, bin_size_y,  bin_size_x],out, on_unused_input = 'warn')
    func1 = function([roi_up, bin_size_y,  bin_size_x],out1, on_unused_input = 'warn')
    '''
    # test 1
    input_roi1 = np.arange(16).reshape(1,4,4).astype(np.float32)
    print input_roi1
    print func(input_roi1,16,16)
    '''
    '''
    # test 2 numerical random number
    input_roi2= np.random.randn(3,500,1000).astype(np.float32)
    #print input_roi2
    import time
    tic = time.clock()
    for i in xrange(1000):
        func(input_roi2, 55,55)
    toc = time.clock()
    print 'WxH -> axH -> axb: '+str(toc-tic)
    tic = time.clock()
    for i in xrange(1000):
        func1(input_roi2, 55,55)
    toc = time.clock()
    print 'WxH -> Wxb -> axb: '+str(toc-tic)
    '''
    
    # test 3 visualized test
    import cv2
    img = cv2.imread('test_small.jpg').transpose(2,0,1) # load color scale test image
    print 'image shape: '+str(img.shape)
    
    out_img = func(img, 1000,1000).transpose(1,2,0).astype(np.uint8)
    out_img1 = func1(img, 1000,1000).transpose(1,2,0).astype(np.uint8)
  
    print out_img.shape, out_img1.shape
    print 'two output images should be the same'
    #cv2.imshow('trans_img',img.transpose(1,2,0))
    cv2.imshow('out_img',out_img)
    cv2.imshow('out_img1',out_img1)
    cv2.waitKey(0)
    
    
