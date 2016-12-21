from keras.preprocessing import image
#from imagenet_utils import preprocess_input
import numpy as np
import os

def add_normalized_bbox(roidb):
    '''normalize roi bbox coordinates to the range [0,1]'''
    assert len(roidb)>0
    num_images = len(roidb)
    for num_i in xrange(num_images):
        rois_orig = roidb[num_i]['boxes'] # dim: nb_rois x 4
        nb_channels, height, width = roidb[num_i]['image_data'].shape 
        rois_normed = np.column_stack(( rois_orig[:,0]/width,rois_orig[:,1]/height,rois_orig[:,2]/width, rois_orig[:,3]/height))
        roidb[num_i]['box_normalized'] = rois_normed 
        
    return 0


def add_image_data(roidb):
    '''load the real image data given its path'''
    assert len(roidb)>0
    num_images = len(roidb)
    for num_i in xrange(num_images):
        img_path = roidb[num_i]['image'] # read image path
        assert os.path.exists(img_path),'image path does not exist: {}'.format(img_path)
        img = image.load_img(img_path)
        img = image.img_to_array(img) # dim: nb_channels,height,width. eg(3,442,500) for 2008_000008.jpg
        #img = np.expand_dims(img, axis=0) # dim:
        # substracting mean rgb pixel intensity computed from image net dataset
        #img = preprocess_input(img)
        roidb[num_i]['image_data'] = img
    return 0
