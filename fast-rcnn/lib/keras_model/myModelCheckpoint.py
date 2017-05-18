import os
import csv
import six

import numpy as np
import time
import json
import warnings
from keras.callbacks import Callback

# unnormalize weights!!!!!!!


class MyModelCheckpoint(Callback):
    '''Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the validation loss will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minization of the monitored. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto',means=None, stds=None):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.means = means
        self.stds = stds

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('MyModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    if (self.means is None) or (self.stds is None):
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        norm_w = self.model.get_layer('bbox_output').get_weights()[0]
                        norm_b = self.model.get_layer('bbox_output').get_weights()[1]
                        self.model.get_layer('bbox_output').set_weights((norm_w* (stds[np.newaxis,:]),norm_b * stds+ means))
                        self.model.save_weights(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if (self.means is None) or (self.stds is None):
                self.model.save_weights(filepath, overwrite=True)
            else:
                stds = self.stds
                means = self.means
                norm_w = self.model.get_layer('bbox_output').get_weights()[0]
                norm_b = self.model.get_layer('bbox_output').get_weights()[1]
                self.model.get_layer('bbox_output').set_weights((norm_w* (stds[np.newaxis,:]),norm_b * stds+ means))
                self.model.save_weights(filepath, overwrite=True)
