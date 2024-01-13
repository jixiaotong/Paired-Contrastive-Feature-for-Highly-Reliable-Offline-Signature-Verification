import cv2
import os
import numpy as np
import random


# normalize each image by dividing the pixel values with the 
# standard deviation of the pixel values of the images in a dataset

#%%

path = '/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/Bengali/'
if_resize = 0
if_skill = 0
if_random = 0
if_skillandrandom = 1

#%%
def resize_img(path, f, resized_dir):
    # threshold = 220
 	try:
         img = cv2.imread(path+'writers/'+str(f)+'/' + path, 0)
         dst = cv2.resize(img, (220, 155), cv2.INTER_LINEAR)
         cv2.imwrite(resized_dir+'{}'.format(path), dst)
         print('############')
 	except:
         print(path)

if if_resize:
    for f in os.listdir(path+'/writers'):
        resized_dir = path+'resized/'+str(f)+'/'
        if not os.path.exists(resized_dir):
            os.makedirs(resized_dir)
        for p in os.listdir(path+'/writers/'+f):
            resize_img(p, f, resized_dir)
       
#%% generate list_all

if if_skill:
    genuine_lines = []
    for gl in open(path+"list.genuine"): 
        genuine_lines.append(gl.strip())
    
    forgery_lines = []
    for fl in open(path+"list.forgery"): 
        forgery_lines.append(fl.strip())
   
    g_c = 0
    f_c = 0
    with open('/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/Bengali/Bengali_pairs.txt', 'w') as f:
        for i in range(100):
            for j in range(24):
                for k in range(j+1, 24):
                    g_c += 1
                    f.write(genuine_lines[i*24+j]+' '+genuine_lines[i*24+k]+' 1\n')
                for l in range(30):
                    f_c += 1
                    f.write(genuine_lines[i*24+j]+' '+forgery_lines[i*30+l]+' 0\n')
               
#%% generate list_random

if if_random:
    genuine_lines = []
    for gl in open(path+"list.genuine"): 
        genuine_lines.append(gl.strip())
   
    g_c = 0 #genuine_count
    f_c = 0 #forgery_count
    with open('/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/Bengali/Bengali_pairs.txt', 'w') as f:
        for i in range(100):
            for j in range(24):
                for k in range(j+1, 24):
                    g_c += 1
                    f.write(genuine_lines[i*24+j]+' '+genuine_lines[i*24+k]+' 1\n')
                for l in range(30):
                    f_c += 1
                    f.write(genuine_lines[i*24+j]+' '+genuine_lines[
                        random.choice([num for num in range(0, 100) if num != i])*24+random.randint(0, 23)]+' 0\n')
                    
#%% generate list_skillandrandom

if if_skillandrandom:
    genuine_lines = []
    for gl in open(path+"list.genuine"): 
        genuine_lines.append(gl.strip())
    
    forgery_lines = []
    for fl in open(path+"list.forgery"): 
        forgery_lines.append(fl.strip())
   
    g_c = 0 #genuine_count
    f_c = 0 #forgery_count
    with open('/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/Bengali/Bengali_pairs.txt', 'w') as f:
        for i in range(100):
            for j in range(24):
                for k in range(j+1, 24):
                    g_c += 1
                    f.write(genuine_lines[i*24+j]+' '+genuine_lines[i*24+k]+' 1\n')
                for l in range(15):
                    f_c += 1
                    f.write(genuine_lines[i*24+j]+' '+forgery_lines[i*30+l]+' 0\n') # skilled
                for l in range(15):  
                    f_c += 1
                    f.write(genuine_lines[i*24+j]+' '+genuine_lines[
                        random.choice([num for num in range(0, 100) if num != i])*24+random.randint(0, 23)]+' 0\n') # random











































































