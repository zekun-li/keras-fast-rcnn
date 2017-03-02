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
from  keras.layers import advanced_activations

def create_yolo_model():
    yolo = Sequential()
    leaky_relu = advanced_activations.LeakyReLU(alpha=0.1) 
    # block 1
    yolo.add(Convolution2D(64,7,7,border_mode = 'same',subsample=(2, 2),input_shape = (3,448,448),name = 'conv1'))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2),name = 'maxpooling1'))
    
    #block 2
    yolo.add(Convolution2D(192,3,3,border_mode = 'same',name = 'conv2'))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2),name = 'maxpooling2'))
    
    #block 3
    yolo.add(Convolution2D(128,1,1,border_mode = 'same',name = 'conv3_1'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,3,3,border_mode = 'same',name = 'conv3_2'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1,border_mode = 'same',name = 'conv3_3'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3,border_mode = 'same',name = 'conv3_4'))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2),name = 'maxpooling3'))
    

    # block 4
    yolo.add(Convolution2D(256,1,1,border_mode = 'same',name = 'conv4_1'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3,border_mode = 'same',name = 'conv4_2'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1,border_mode = 'same',name = 'conv4_3'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3,border_mode = 'same',name = 'conv4_4'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1,border_mode = 'same',name = 'conv4_5'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3,border_mode = 'same',name = 'conv4_6'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1,border_mode = 'same',name = 'conv4_7'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3,border_mode = 'same',name = 'conv4_8'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,1,1,border_mode = 'same',name = 'conv4_9'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv4_10'))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2),name = 'maxpooling4'))
    
    # block 5
    yolo.add(Convolution2D(512,1,1,border_mode = 'same',name = 'conv5_1'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv5_2'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,1,1,border_mode = 'same',name = 'conv5_3'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv5_4'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv5_5'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',subsample=(2, 2),name = 'conv5_6'))
    yolo.add(leaky_relu)
  
    # block 6
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv6_1'))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3,border_mode = 'same',name = 'conv6_2'))
    yolo.add(leaky_relu)
    
    # block 7 fc layer
    yolo.add(Flatten(name = 'flatten'))
    yolo.add(Dense(4096, name = 'fc1'))
    yolo.add(leaky_relu)
    yolo.add(Dropout(0.5))
    
    # block 8 fc layer
    yolo.add(Dense(1470,activation = 'linear',name = 'fc2'))
    
    # reshape output from 1D to 3D
    yolo.add(Reshape((30,7,7)))
    
    return yolo

# global variables for loss function
lambda_coord = 5
lambda_noobj = 0.5

# target: 3D, S^2 x Boxes x (4 coords + 20 classes + objectness) ?
def multi_part_loss(target,predicted):
    x_target = target[:,:,0]
    y_target = target[:,:,1]
    x_predicted = predicted[:,:,0]
    y_predicted = predicted[:,:,0]
    

if __name__ == "__main__":
    # construct yolo network
    input_x = Input( shape = ( 3,448 ,448 ), name = 'batch_of_images' )
    model = create_yolo_model()
    output_y = model(input_x)
    yolo = Model(input = [input_x], output = [output_y])
    yolo.summary()

    # TODO: implement loss function
    yolo.compile(optimizer = 'sgd', loss= 'categorical_crossentropy', metrics = ['accuracy'])

    print model.get_layer('maxpooling1').output_shape
    print model.get_layer('maxpooling2').output_shape
    print model.get_layer('maxpooling3').output_shape
    print model.get_layer('maxpooling4').output_shape
    print model.get_layer('conv5_6').output_shape
    print model.get_layer('conv6_2').output_shape
