# -*- coding: utf-8 -*-

import os

dataset = 'Bengali'
iftrain = 1
for i in range(1,2): # cross-validation
    for j in range(1,11): # trial
        print('###### Fold=%s, Trial=%s, Iftrain=%s ######' %(i,j,iftrain))
        if  os.path.exists('/home/xiaotong/Desktop/Experiments/Codes/Reliable_PCF/Top_rankNN/TopRankNN_models/'+dataset+'/CV'+str(i)+'_'+str(j)+'/p_0'):
            print('TopRankNN trained on this trial=%s' %j)
            continue    
        print('Training on trial=%s' %j)
        os.system("python TopRankNN.py "+dataset+" %s %s %s" %(i,j,iftrain))