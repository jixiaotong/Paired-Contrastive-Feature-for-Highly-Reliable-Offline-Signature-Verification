# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from SignatureDataGenerator import *

#%% MODE STRUCTURE

def deepnn(x, train):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 2048])
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([2048, 1024])
    b_fc1 = bias_variable([1024]) 
    h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)
    
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512]) 
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([512, 128])
    b_fc3 = bias_variable([128]) 
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)    
    
  with tf.name_scope('fc4'):
    W_fc4 = weight_variable([128, 1])
    b_fc4 = bias_variable([1])
    y_conv = tf.sigmoid(tf.matmul(h_fc3, W_fc4) + b_fc4)
    return y_conv, h_fc3


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#%% GET BATCH

def get_batch(name_pos, label_pos, name_neg, label_neg, batch_size_p, batch_size_n, now_batch_p, now_batch_n, total_batch_n):
    image_batch_pos = name_pos[now_batch_p*batch_size_p:(now_batch_p+1)*batch_size_p,:]
    label_batch_pos = label_pos[now_batch_p*batch_size_p:(now_batch_p+1)*batch_size_p]
    image_batch_neg = name_neg[now_batch_n*batch_size_n:(now_batch_n+1)*batch_size_n,:]
    label_batch_neg = label_neg[now_batch_n*batch_size_n:(now_batch_n+1)*batch_size_n]        
    batchdata=np.concatenate((image_batch_pos,image_batch_neg),axis=0)        
    batchlabel=np.concatenate((label_batch_pos,label_batch_neg),axis=0)
    return batchdata, batchlabel


#%% TOP-RANK LOSS

def loss_with_top_rank(logits, labels, p):
    
  with tf.name_scope('toprank'):
      
      loss = tf.zeros([1,1])           
      num = tf.constant(1/5)            
      index1 = tf.where(tf.equal(labels,1))      
      indices1 = index1[:,0]            
      index2 = tf.where(tf.not_equal(labels,1))        
      indices2 = index2[:,0]         
      cs=tf.constant(1.0)
        
      p1 = p     
      p2 = cs/p1
            
      sum_n = tf.zeros([1,1])          
      sum_p = tf.zeros([1,1])
            
      l = tf.zeros([1,1])
            
      feature = tf.zeros([1,1])  
      norm_feature = tf.zeros([1,1])
            
      l_p = tf.zeros([1,1])
            
      for u in range(5): # pos_num    
          for v in range(40): # neg_num
              feature = tf.subtract(logits[indices1[u]],logits[indices2[v]])
              norm_feature =tf.log(1+tf.exp(-feature))
              l_p = tf.pow(norm_feature,p1)
              sum_n = sum_n+l_p
    
          l = tf.pow(sum_n,p2)
          sum_p = sum_p+l
              
      loss = tf.multiply(sum_p,num)         
          
      return loss
  