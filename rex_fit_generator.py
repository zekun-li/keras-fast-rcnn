import cv2
import numpy as np 

def convert_roidb_image_to_net_input( roidb_image ) :
    
    ch, h, w = roidb_image.shape

def compress( image_array ) :
    _, img_buf = cv2.imencode( 'tmp.jpg', image_array.astype('uint8') )
    return img_buf

def decompress( image_buf ) :
    return cv2.imdecode( image_buf, -1 )

def convert_roidb_elem_to_net_inputs( roidb_elem, use_compression = False ) :
    cls_target  = get_cls_target( roidb_elem ).astype( np.int32 )
    dbox_target = get_delta_box_target( roidb_elem ).astype( np.float32 )
    roi_input = get_roi_input( roidb_elem ).astype( np.float32 )
    # get image input
    resized_image = get_image_input( roidb_elem ).astype( 'uint8' )
    if ( use_compression ) :
        img_buf = compress( resized_image )
        return ( img_buf, roi_input, cls_target, dbox_target )
    else :
        return ( resized_image, roi_input, cls_target, dbox_target )

def data_generator( data_points, mode = 'training', use_compression = True, batch_size = 1, nb_epoch = -1 ) :
    def prepare_one_sample( data_point ) :
        img, roi, cls, dbox = data_point
        if ( use_compression ) :
            X = decompress( img )
        else :
            X = img
        X = X.astype( np.float32 )
        # match normalization of FAST RCNN
        Xn = normalize( X )
        X_tensor = np.expand_dims( np.rollaxis( Xn, 2, 0 ), axis = 0 )
        return { 'x4d' : X_tesnor, 'roi3d' : roi }, { 'output_1' : cls, 'output_2' : dbox }
    indices = range( len( data_points ) )
    epoch = 0
    while ( epoch < nb_epoch ) or ( nb_epoch <=0 ) :
        idx = np.random.choice( indices )
        data_point = data_point[idx]
        model_input, model_output = prepare_one_sample( data_point )
        yield model_input, model_output
    return

use_compression = True
data_points = [ convert_roidb_elem_to_net_inputs( elem, use_compression ) in roidb ]
