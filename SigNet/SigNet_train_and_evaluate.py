# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import joblib
from natsort import natsorted
import gc
from tensorflow.compat.v1.keras import backend as K
import numpy as np
# from SigNet_structure import SigNet
from SigNet_structure import *
import matplotlib.pyplot as plt
import os
import psutil
import sys 

lrs = [2e-5, 5e-5, 1e-4]
maxepcs = [60, 50, 40]

#%% if train
if di.iftrain:
    for learning_rates, max_epoches in zip(lrs, maxepcs):
        K.clear_session()
        tf.reset_default_graph()
        
        file = './SigNet_models/'+str(di.dataset)+'/CV'+str(di.cv)+'_'+str(di.trial)+'/CV'+str(di.cv)+'_'+str(di.trial)+'_'+str(learning_rates)+'/'
        if not os.path.exists(file):
            os.makedirs(file)
        model = SigNet(train=True, filename=file, batch_size=256, max_epoches=max_epoches, modelname='', lr=learning_rates)
        del model
        gc.collect()
            
    #--------------------------------------------------------
    # record accuracy, 1 by 1 due to memory limit   
    for learning_rates, max_epoches in zip(lrs, maxepcs):
        file = './SigNet_models/'+str(di.dataset)+'/CV'+str(di.cv)+'_'+str(di.trial)+'/CV'+str(di.cv)+'_'+str(di.trial)+'_'+str(learning_rates)+'/'
        print('***%s***' %file)
        for root, dirs, files in os.walk(file):
            for i in natsorted(files):
              if '.hdf5' in i and 'pred_and_acc_' not in i and 'pred_and_acc_'+i[:-5] not in files: # and int(i.split('-')[0].split('.')[1])>10:
                    K.clear_session()
                    tf.reset_default_graph()
                    print('***%s***' %i)
                    dicti = {}
                    model = SigNet(train=False, filename=file, batch_size=128, \
                                    max_epoches=max_epoches, modelname=i)
                    train_pred = model.predict(model.train_images, model.model, 128)
                    print('train_pred')
                    train_acc, dd = model.calculate_accuracy_roc(train_pred, model.train_labels, train_pred, model.train_labels, 0)
                    print('train_acc')
                    print(dd)
                    train_loss = model.calculate_loss(train_pred, model.train_labels)
                    print('train_loss')
                    dicti['train'] = {'acc': train_acc, 'pred': train_pred, 'cla_thre': dd, 'loss': train_loss}
                    del train_pred
                    gc.collect()
                    valid_pred = model.predict(model.valid_images, model.model, 128)
                    valid_acc, _ = model.calculate_accuracy_roc([], [], valid_pred, model.valid_labels, dd)
                    valid_loss = model.calculate_loss(valid_pred, model.valid_labels)
                    dicti['valid'] = {'acc': valid_acc, 'pred': valid_pred, 'cla_thre': dd, 'loss': valid_loss}
                    del valid_pred
                    gc.collect()
                    test_pred = model.predict(model.test_images, model.model, 128)
                    test_acc, _ = model.calculate_accuracy_roc([], [], test_pred, model.test_labels, dd)
                    test_loss = model.calculate_loss(test_pred, model.test_labels)
                    dicti['test'] = {'acc': test_acc, 'pred': test_pred, 'cla_thre': dd, 'loss': test_loss}
                    joblib.dump(dicti, './SigNet_models/'+str(di.dataset)+'/CV'+str(di.cv)+'_'+str(di.trial)+'/CV'+str(di.cv)+'_'+str(di.trial)+'_'+str(learning_rates)+'/pred_and_acc_'+i[:-5])
                    del test_pred
                    del dicti
                    del model
                    gc.collect()
           
    #--------------------------------------------------------
    # concatenate all accuracies
    
    import joblib
    import os
    import numpy as np
    from natsort import natsorted
    
    for learning_rates, max_epoches in zip(lrs,maxepcs):
        conca = []
        file = './SigNet_models/'+str(di.dataset)+'/CV'+str(di.cv)+'_'+str(di.trial)+'/CV'+str(di.cv)+'_'+str(di.trial)+'_'+str(learning_rates)+'/'
        for root, dirs, files in os.walk(file):
            for i in natsorted(files):
                if 'pred_and_acc' in i and 'all' not in i:
                    conca.append(joblib.load(file+i))
        joblib.dump(conca, file+'all_pred_and_acc')


#%% accuracy
elif di.iftrain == 0:

    best_conca_valid_accs = []
    best_conca_test_accs = []
    best_learnings = []
    best_indxs = []
    for ditrial in range(2,3):
        best_learning_rate = 0
        best_conca_valid_acc = 0
        best_conca_test_acc = 0
        best_learning = 0
        best_indx = 0
        if not os.path.exists('./SigNet_models/'+str(di.dataset)+'/CV'+str(di.cv)+'_'+str(di.trial)):
            print('trial%s is empty yet' %di.trial)
            best_conca_valid_accs.append(0)
            best_learnings.append(0)
            best_indxs.append(0)
            continue
        for learning_rates in lrs:
            file = './SigNet_models/'+str(didataset)+'/CV'+str(di.cv)+'_'+str(di.trial)+'/CV'+str(di.cv)+'_'+str(di.trial)+'_'+str(learning_rates)+'/'
            if not os.path.exists(file+'all_pred_and_acc'):
                print('trial%s has not all yet' %ditrial)
                continue
            conca = joblib.load(file+'all_pred_and_acc')
            fig = plt.figure(figsize=(10, 6), dpi=120)
            plt.plot([conca[i]['train']['acc'] for i in range(len(conca))], linewidth=2, label='train_acc', color='#716e77', linestyle='-')
            plt.plot([conca[i]['valid']['acc'] for i in range(len(conca))], linewidth=2, label='valid_acc', color='#f9b69c', linestyle='-')
            plt.plot([conca[i]['test']['acc'] for i in range(len(conca))], linewidth=2, label='test_acc', color='#cd5554', linestyle=':')
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.grid()
            plt.legend(['train_acc', 'valid_acc', 'test_acc'], loc='lower right')
            plt.show()
            
            index_good_train_acc = [j for j in range(len(conca)) if conca[j]['train']['acc']>0.7 and conca[j]['train']['acc']<0.99]
            # print('--- good valids for lr=%s ---' %learning_rates)
            for i in index_good_train_acc:
                # print('index=%s, valid_acc=%s' %(i, conca[i]['valid']['acc']))
                if conca[i]['valid']['acc']>best_conca_valid_acc and conca[i]['valid']['acc']<0.99:                
                    best_conca_valid_acc = conca[i]['valid']['acc']
                    best_conca_test_acc = conca[i]['test']['acc']
                    best_learning = learning_rates
                    best_indx = i
        # print('---------------------------------')
        best_conca_valid_accs.append(best_conca_valid_acc)
        best_conca_test_accs.append(best_conca_test_acc)
        best_learnings.append(best_learning)
        best_indxs.append(best_indx)

    print('# best learning_Rates=%s' %best_learnings)
    print('# best conca_valid_accs=%s' %best_conca_valid_accs)
    print('# best indexs=%s' %best_indxs) 
        
    print('# best conca_test_accs=%s' %best_conca_test_accs)
    
