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
    yolo.add(Convolution2D(64,7,7,input_shape=(3,448,448)))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2)))
    
    #block 2
    yolo.add(Convolution2D(192,3,3))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2)))
    
    #block 3
    yolo.add(Convolution2D(128,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,3,3))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2)))
    

    # block 4
    yolo.add(Convolution2D(256,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(256,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,3,3))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(512,1,1))
    yolo.add(leaky_relu)
    yolo.add(Convolution2D(1024,3,3))
    yolo.add(leaky_relu)
    yolo.add(MaxPooling2D((2,2),strides = (2,2)))

    
             
    return yolo
