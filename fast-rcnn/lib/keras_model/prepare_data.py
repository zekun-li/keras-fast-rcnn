from keras.preprocessing import image
import numpy as np
import os


# substract mean from each channel
# switch channel order
def preprocess(x):
    # assume dim_ordering is th, can use keras.backend.image_dim_ordering() to check
    x[0, :, :] -= 103.939
    x[1, :, :] -= 116.779
    x[2, :, :] -= 123.68
    # 'RGB'->'BGR'
    x = x[::-1, :, :]
    return x


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
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) # dim: nb_channels,height,width. eg(3,442,500) for 2008_000008.jpg
        # substracting mean rgb pixel intensity computed from image net dataset
        # also switch channels
        img = preprocess(img)
        roidb[num_i]['image_data'] = img
    return 0
