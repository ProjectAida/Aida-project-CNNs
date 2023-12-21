#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import TensorFlow.

import IPython.display as display
import numpy as np
import os
import datetime
import cv2

import tensorflow as tf
from tensorflow.keras.utils import Sequence

import keras
from keras import backend as K
from keras.callbacks import Callback
from keras import metrics

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from model_factory import GetModel

import multiprocessing

import sys

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, augs = [], base_path = '',
                 to_fit=True, batch_size=50, dim=(192, 128),
                 n_channels=1, n_classes=16, shuffle=False, last_batch='keep'):
        """Initialization

        :param list_IDs: list of all 'label' ids to use in the generator
        :param list_IDs: base path of the images
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        :param last_batch: either 'keep', 'loop', 'discard' for the last batch
        """
        self.list_IDs = list_IDs
        self.augs = augs
        self.base_path = base_path
        self.total_nb = len(list_IDs)
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.last_batch = last_batch
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        if(self.last_batch == 'discard'):
            self.nb_batch = int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            self.nb_batch = int(np.ceil(len(self.list_IDs) / self.batch_size))
            
        return self.nb_batch

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        if(((index + 1) * self.batch_size) >= self.total_nb):
            if(self.last_batch == 'keep'):
                indexes = self.indexes[index * self.batch_size:self.total_nb]
            if(self.last_batch == 'loop'):
                indexes = self.indexes[index * self.batch_size:self.total_nb]
                indexes = np.append(indexes,self.indexes[0:self.batch_size-len(indexes)])
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_images(list_IDs_temp)

        if self.to_fit:
            y = self._generate_labels(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(self.total_nb)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_images(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))

        # Generate data
        for i, line in enumerate(list_IDs_temp):
            # get image path
            img_path = line.split(',')[0]
            #img_path = img_path.replace('.tif', '.png')
            # Store sample
            X[i,] = self._load_image(os.path.join(self.base_path, img_path))

        return X

    def _generate_labels(self, list_IDs_temp):
        """Generates data containing batch_size masks

        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty(len(list_IDs_temp), dtype=int)

        # Generate data
        for i, line in enumerate(list_IDs_temp):
            # get label
            img_label = line.split(',')[1]
            # Store sample
            y[i,] = img_label
        return y

    def _load_image(self, image_path):
        """Load grayscale image

        :param image_path: path to image to load
        :return: loaded image
        """
#         print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         if(np.prod(img.shape) * self.n_channels == np.prod(self.dim) * self.n_channels):
#             img=np.stack((img,)*self.n_channels, axis=-1)
        
        for aug in self.augs:
            img = aug(img)
            
        img = np.stack((img,), axis=-1)
        
        img = img / 255
        return img

def run_train(train_datalist, val_datalist, base_path, nClass, mod_name, nEpoch):
    
    tf.enable_resource_variables()
    tf.logging.set_verbosity(tf.logging.ERROR)
    N_CPUS = multiprocessing.cpu_count()

    height = 192
    width = 128
    def Resize(img):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img
    augs = [Resize]

    model = GetModel(mod_name, nClass, height=height, width=width)

    bset_f1 = 0.0

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'] #metrics=metrics
                 )

    # test_x = np.load(base_path+'test_bin_cons_img.npy')
    # test_y = np.load(base_path+'test_bin_cons_lbl.npy')

    # print(test_x.shape)
    # print(test_y.shape)
    training_generator = DataGenerator(train_datalist, augs, base_path)
    validation_generator = DataGenerator(val_datalist, augs, base_path)

    train_history = model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator, 
                                 validation_steps=100, 
                                 callbacks=[earlystop], #callbacks=[callback, earlystop],
                                 workers=N_CPUS, 
                                 epochs=nEpoch)

    return train_history, model

def run_eval(test_datalist, test_labels, model_path, base_path, nClass, mod_name):
    
    height = 192
    width = 128
    def Resize(img):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img
    augs = [Resize]
    
    model = GetModel(mod_name, nClass, height=height, width=width)
    model.load_weights(model_path)
    
    testing_generator = DataGenerator(test_datalist, augs, base_path)
    test_preds = model.predict_generator(testing_generator, verbose=1)
    test_preds = np.argmax(test_preds, axis=-1)
    
    conf_matrix = confusion_matrix(test_labels, test_preds)
    
    precision=np.nan_to_num(np.diagonal(conf_matrix)/np.sum(conf_matrix, axis=0))
    recall = np.nan_to_num(np.diagonal(conf_matrix)/np.sum(conf_matrix, axis=-1))
    f1 = np.nan_to_num((2*precision*recall)/(precision+recall))
    acc = np.diagonal(conf_matrix).sum() / conf_matrix.sum()

    return test_preds, conf_matrix, precision, recall, f1, acc

def run_test(test_datalist, model_path, base_path, nClass, mod_name):
    
    height = 192
    width = 128
    def Resize(img):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img
    augs = [Resize]

    model = GetModel(mod_name, nClass, height=height, width=width)
    model.load_weights(model_path)
    
    testing_generator = DataGenerator(test_datalist, augs, base_path, to_fit=False)
    test_preds = model.predict_generator(testing_generator, verbose=1)
    test_preds = np.argmax(test_preds, axis=-1)

    return test_preds

def run_finetune(train_datalist, val_datalist, model_path, base_path, nClass, mod_name, nEpoch):
    
    tf.enable_resource_variables()
    tf.logging.set_verbosity(tf.logging.ERROR)
    N_CPUS = multiprocessing.cpu_count()

    height = 192
    width = 128
    def Resize(img):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img
    augs = [Resize]

    model = GetModel(mod_name, nClass, height=height, width=width)
    model.load_weights(model_path)

    bset_f1 = 0.0

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'] #metrics=metrics
                 )

    # test_x = np.load(base_path+'test_bin_cons_img.npy')
    # test_y = np.load(base_path+'test_bin_cons_lbl.npy')

    # print(test_x.shape)
    # print(test_y.shape)
    training_generator = DataGenerator(train_datalist, augs, base_path)
    validation_generator = DataGenerator(val_datalist, augs, base_path)
#     testing_generator = DataGenerator(test_datalist, augs, base_path)

    train_history = model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator, 
                                 validation_steps=100, 
                                 callbacks=[earlystop], #callbacks=[callback, earlystop],
                                 workers=N_CPUS, 
                                 epochs=nEpoch)

    return train_history, model