# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import joblib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1.keras import backend as K
import getpass as gp
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import random
random.seed(1234)
np.random.seed(1234)

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
import sys 
sys.path.append("..") 

# from SignatureDataGenerator import SignatureDataGenerator
from SignatureDataGenerator import *

#%%

class SigNet(object):
    K.clear_session()
    tf.reset_default_graph()
    def __init__(self, train=True, filename='', batch_size=128, max_epoches=30, modelname='', lr=1e-5):
        print('---Generating SigNet---')
        self.train_images = 0
        self.train_labels = 0
        self.valid_images = 0
        self.valid_labels = 0
        self.test_images = 0
        self.test_labels = 0
        self.learning_rate = lr
        self.modelname = modelname
        self.batch_size = batch_size
        self.datagen = 0
        self.std = 0
        self.max_epoches = max_epoches
        self.cv = di.cv
        self.trial = di.trial
        self.input_shape=(di.img_height, di.img_width, 1)
        self.filename = filename
        self.model, self.model_features = self.build_model(self.input_shape, Input(shape=self.input_shape), Input(shape=self.input_shape))
        # self.model.summary()
        self._load_data(train)
        if train:
            print('---[Train] is True---')            
            self.model = self.train(self.model)
        else:
            print('---[Train] is False---')
            self.model.load_weights(self.filename+'/'+self.modelname)
            print('---model loaded---')
     
    #----------------------------------------------------------------
    def euclidean_distance(self,vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
            
    def eucl_dist_output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
        
    #--------------------------------------------------------------------       
    def SigNet_single(self, input_shape):
        input_tensor = Input(shape=input_shape)
        x = Conv2D(96, (11, 11), activation='relu', name='conv1_1', strides=(4, 4), \
                   kernel_initializer='glorot_uniform')(input_tensor)
        x = BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)(x)
        x = MaxPooling2D((3,3), strides=(2, 2), padding='same')(x) 
        x = ZeroPadding2D((2, 2))(x)
        
        x = Conv2D(256, (5, 5), activation='relu', name='conv2_1', strides=(1, 1), \
                   kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9)(x)
        x = MaxPooling2D((3,3), strides=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)
        x = ZeroPadding2D((1, 1))(x)
        
        x = Conv2D(384, (3, 3), activation='relu', name='conv3_1', strides=(1, 1), \
                   kernel_initializer='glorot_uniform')(x)
        x = ZeroPadding2D((1, 1))(x)
        
        conv = Conv2D(256, (3, 3), activation='relu', name='conv3_2', strides=(1, 1), \
                      kernel_initializer='glorot_uniform')(x)
        x = MaxPooling2D((3,3), strides=(2, 2), padding='same')(conv)
        x = Dropout(0.3)(x)
        #    model.add(SpatialPyramidPooling([1, 2, 4]))
        x = Flatten(name='flatten')(x)
        fc = Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.5)(fc)
        x = Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')(x)
        
        return Model(inputs=input_tensor, outputs=x), Model(inputs=input_tensor, outputs=fc)

    #--------------------------------------------------------------------
    def build_model(self, x_shape, input_a, input_b):
        print('---Building SigNet model---')        
        base_network, shallow_network = self.SigNet_single(x_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)    
        shallow_a = shallow_network(input_a)
        shallow_b = shallow_network(input_b)    

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model(inputs=[input_a, input_b], outputs=distance)
        model_ = Model(inputs=[input_a, input_b], outputs=(shallow_a, shallow_b))
        return model, model_
    
    #------------------------------------------------------------------------    
    def standardize(self, x, std):
        x /= (std + 1e-7)
        return x

    def read_image_and_labels(self, lines, height, width, image_dir, std):
        image_pairs = []
        label_pairs = []
        images = []                
        for line in lines:
            file1, file2, label = line.split(' ')
            
            img1 = image.load_img(image_dir + file1, grayscale = True,
            target_size=(height, width))
                            
            img1 = image.img_to_array(img1)
            if self.std!=0:
                img1 = self.standardize(img1, self.std)
            images.append(img1)
                    
            img2 = image.load_img(image_dir + file2, grayscale = True,
            target_size=(height, width))
            
            img2 = image.img_to_array(img2)
            if self.std!=0:
                img2 = self.standardize(img2, self.std)
            images.append(img1)
            
            image_pairs += [[img1, img2]]
            label_pairs += [int(label)]
        image_pairs = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
        label_pairs = np.array(label_pairs)
        return image_pairs, label_pairs, np.array(images)

    def _load_data(self, train):
        # train_lines, valid_lines, test_lines = self.read_lines(di.tot_writers, di.num_train_writers, di.num_valid_writers, di.num_test_writers, di.feature_extraction_number, '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/Bengali/gray_all.txt')
        datagen = SignatureDataGenerator(di.cv, di.dataset, di.tot_writers, di.num_train_writers, 
        di.num_valid_writers, di.num_test_writers, di.nsamples, self.batch_size, di.img_height, di.img_width,
        di.featurewise_center, di.featurewise_std_normalization, di.zca_whitening)
        self.datagen = datagen
        if di.dataset=='Bengali' or di.dataset=='Hindi':
            direc = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/'+di.dataset+'/resized/'
            _, _, trainingset = self.read_image_and_labels(datagen.train_lines, 155, 220, direc, 0)
        elif di.dataset=='Chinese' or di.dataset=='Dutch':
            direc = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/trainingSet/OfflineSignatures/'+di.dataset+'/TrainingSet/'
            direc_test = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/Testdata_SigComp2011/SigComp11-Offlinetestset/'+di.dataset+'/'
            _, _, trainingset = self.read_image_and_labels(datagen.train_lines, 155, 220, direc, 0)         
        elif di.dataset=='UTSig':
            direc = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/'+di.dataset+'/'
            _, _, trainingset = self.read_image_and_labels(datagen.train_lines, 155, 220, direc, 0)         
        std = datagen.fit(trainingset)
        self.std = std
        
        if not train:
            if di.dataset=='Bengali' or di.dataset=='Hindi':
                ro_dir = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/'+di.dataset+'/'
                # data_file = ro_dir + 'gray_all.txt'
                data_file = ro_dir + 'Bengali_pairs.txt'
            elif di.dataset=='Chinese' or di.dataset=='Dutch':
                ro_dir = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/trainingSet/OfflineSignatures/'+di.dataset+'/TrainingSet/'
                data_file = ro_dir + 'gray_all.txt'
            elif di.dataset=='UTSig':
                ro_dir = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/'+di.dataset+'/'
                data_file = ro_dir + 'gray_all.txt'
            f = open( data_file, 'r' )
            lines = f.readlines()
            f.close()  
            self.train_images, self.train_labels, _ = self.read_image_and_labels(datagen.train_lines, 155, 220, direc, self.std)  
            
            self.valid_images, self.valid_labels, _ = self.read_image_and_labels(datagen.valid_lines, 155, 220, direc, self.std)    

            if not di.dataset=='Chinese' and not di.dataset=='Dutch':
                self.test_images, self.test_labels, _ = self.read_image_and_labels(datagen.test_lines, 155, 220, direc, self.std)                       
            if di.dataset=='Chinese' or di.dataset=='Dutch':
                self.test_images, self.test_labels, _ = self.read_image_and_labels(datagen.test_lines, 155, 220, direc_test, self.std)                       

    #-------------------------------------------------------------------------
    def train(self, model):          
        print('---Starting training---')
        
        def contrastive_loss(y_true, y_pred):
            margin = 1
            return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
        def compute_accuracy_roc(y_true, y_pred):
            dmax = K.cast(K.max(y_pred), K.floatx())
            dmin = K.cast(K.min(y_pred), K.floatx())
            nsame = K.cast(K.sum(K.cast(K.greater(y_true, 0.5), K.floatx())), K.floatx())
            ndiff = K.cast(K.sum(K.cast(K.greater(0.5, y_true), K.floatx())), K.floatx())
            
            step = 0.01
            max_acc = 0.0
            
            for d in K.arange(dmin, dmax+step, step):
                idx1 = K.greater_equal(d, y_pred)
                idx2 = K.greater(y_pred, d)
                
                tpr = K.cast(K.sum(K.cast(K.greater(K.cast(y_true[idx1], K.floatx()), 0.5), K.floatx())), K.floatx()) / nsame+0.00000001       
                tnr = K.cast(K.sum(K.cast(K.greater(0.5, K.cast(y_true[idx2], K.floatx())), K.floatx())), K.floatx()) / ndiff+0.00000001
                out = Lambda(lambda a: a[0] + a[1])([tpr, tnr])
                acc = Lambda(lambda x: x * 0.5)(out) 
                
                if K.greater(acc, max_acc):
                     max_acc = acc                  
            return max_acc
       
        rms = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08)
        model.compile(optimizer=rms, loss=contrastive_loss)#, metrics=[compute_accuracy_roc])#{'lambda':compute_accuracy_roc})
        print('---model compiled---')
        
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=self.filename+'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)
        historytemp = model.fit_generator(self.datagen.next_train(),
                                          steps_per_epoch=2*di.nsamples*di.num_train_writers // self.batch_size,
                                          epochs=self.max_epoches, callbacks=[model_checkpoint_callback], workers=4,
                                            use_multiprocessing=True, validation_data=self.datagen.next_valid(), 
                                            validation_steps=2*di.nsamples*di.num_valid_writers // self.batch_size)
        joblib.dump(historytemp.history, self.filename+'history')
        fig = plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(historytemp.history['loss'], linewidth=2, label='training loss', color='#716e77', linestyle='-')
        plt.plot(historytemp.history['val_loss'], linewidth=2, label='validation loss', color='#f9b69c', linestyle='-')
        plt.title('Model losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['training loss', 'validation loss'], loc='upper right')
        plt.show()

        return model
    
    #--------------------------------------------------------------------------
    def predict(self, x=None, model=None, batch_size=128):
        if x is None:
            x = self.x_test
        return model.predict(x, batch_size)
    
    def calculate_accuracy_roc(self, train_pred, train_true, pred, true, dd):
        
        if dd==0 and not len(train_pred)==0:
            dmax = np.max(train_pred)
            dmin = np.min(train_pred)
            nsame = np.sum(train_true == 1)
            ndiff = np.sum(train_true == 0)
            
            step = 0.01
            max_acc = 0
            dd = 0
           
            for d in np.arange(dmin, dmax+step, step):
                idx1 = train_pred.ravel() <= d
                idx2 = train_pred.ravel() > d
                
                tpr = float(np.sum(train_true[idx1] == 1)) / nsame       
                tnr = float(np.sum(train_true[idx2] == 0)) / ndiff
                acc = 0.5 * (tpr + tnr)   
           
                if (acc > max_acc):
                    max_acc = acc
                    dd = d
            accu = max_acc
            
        else:
            dmax = np.max(pred)
            dmin = np.min(pred)
            nsame = np.sum(true == 1)
            ndiff = np.sum(true == 0)
            
            idx1 = pred.ravel() <= dd
            idx2 = pred.ravel() > dd
            
            tpr = float(np.sum(true[idx1] == 1)) / nsame       
            tnr = float(np.sum(true[idx2] == 0)) / ndiff
            accu = 0.5 * (tpr + tnr)   
           
        return accu, dd      
    
    def calculate_loss(self, pred, true):
        margin = 1
        return np.mean(np.multiply(true, (pred.reshape(len(pred)))**2) + np.multiply((1 - true), (np.array([max(i, 0) for i in margin-pred.reshape(len(pred))])**2)))
                   
