# -*- coding: utf-8 -*-

import numpy as np

#%%
class datasets_info(object):
    def __init__(self, dataset, cv, trial, ini, iftrain='False'):
        self.dataset = dataset
        self.cv = cv
        self.trial = trial
        self.ini = ini
        self.iftrain = iftrain
    # ini = "he_uniform"        
        if dataset == 'Bengali':
            self.tot_writers = 100
            self.num_train_writers = int((self.tot_writers/10)*8)
            self.num_valid_writers = int((self.tot_writers/10)*1)
            self.accuracy_number = 552
            self.feature_extraction_number = 996
            self.nsamples = 276
            self.best_lr = []
            self.best_idx = []
            self.best_thre = {}
            
        elif dataset == 'Hindi':        
            self.tot_writers = 160
            self.num_train_writers = int((self.tot_writers/10)*8)
            self.num_valid_writers = int((self.tot_writers/10)*1)
            self.accuracy_number = 552
            self.feature_extraction_number = 996
            self.nsamples = 276
            self.best_lr = []
            self.best_idx = []
            self.best_thre = {}

        elif dataset == 'UTSig':        
            self.tot_writers = 110
            self.num_train_writers = int((self.tot_writers/10)*8)
            self.num_valid_writers = int((self.tot_writers/10)*1)
            self.accuracy_number = 702
            self.feature_extraction_number = 1566
            self.nsamples = 351
            self.best_lr = []
            self.best_idx = []
            self.best_thre = {}
    
        self.num_test_writers = self.tot_writers - (self.num_train_writers + self.num_valid_writers)
        self.featurewise_center = False
        self.featurewise_std_normalization = True
        self.zca_whitening = False
        self.img_height = 155
        self.img_width = 220    
        self.zca_whitening = False

