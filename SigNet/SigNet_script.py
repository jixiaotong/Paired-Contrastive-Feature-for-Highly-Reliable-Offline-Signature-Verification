# -*- coding: utf-8 -*-

import os

dataset = 'Hindi'
iftrain = 1
for i in range(1,2): # cross-validation
    for j in range(1,2): # trial
        print('###### Fold=%s, Trial=%s, Iftrain=%s ######' %(i,j,iftrain))
        if  os.path.exists('/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/SigNet/SigNet_models/'+dataset+'/CV'+str(i)+'_'+str(j)+'/p_0'):
            print('TopRankNN trained on this trial=%s' %j)
            continue    
        print('Training on trial=%s' %j)
        os.system("python SigNet_train_and_evaluate.py "+dataset+" %s %s %s" %(i,j,iftrain))