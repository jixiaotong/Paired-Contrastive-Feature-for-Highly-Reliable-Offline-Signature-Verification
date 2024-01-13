# -*- coding: utf-8 -*-

import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import joblib
import sys 
sys.path.append("..") 
import Functions
from SignatureDataGenerator import *
  
#%%

def main(di_dataset, di_cv, di_trial):   
    print(di_dataset)
    print(di_cv)
    print(di_trial)
    di_cv = int(di_cv)
    di_trial = int(di_trial)
    
    for fold in range(6):
        p_array=[2,4,8,16,32,64]
        num_p=p_array[fold]

        p_value=np.float32(num_p)
        p_value=np.reshape(p_value,[1])
        tf.reset_default_graph()
        model_name = di_dataset+'_model'+'_p_' + str(num_p)
        training = tf.placeholder(tf.bool)    
        x = tf.placeholder(tf.float32, [None, 2048])
        y_ = tf.placeholder(tf.int64, [None,1])
        y_conv,fc1 = Functions.deepnn(x, training)

        with tf.name_scope('loss'):
    
            toprankloss = Functions.loss_with_top_rank(logits=y_conv, labels=y_, p=p_value)
    
            toprankloss = tf.reshape(toprankloss,[])

            initial_loss= tf.constant(1e-14)

            toprankloss = tf.add(toprankloss,initial_loss)

            loss_summary = tf.summary.scalar('loss', toprankloss)      

        global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                            global_step,
                                            decay_steps=1,decay_rate=0.90)
        with tf.name_scope('adam_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(toprankloss)
        add_global = global_step.assign_add(1)

        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()                  
        step=0
        for epoch in range(50):
            # Here, it is essential to commence by training a feature extractor (such as a SigNet I used).
            # Subsequently, utilize the trained feature extractor to extract signature features and store the resultant outputs.
            # The following code snippets are designed for loading the saved features and labels.
            rootdir = '/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/SigNet/SigNet_models/'+di_dataset+'/'  
            for root, dire, file in os.walk(rootdir+'CV1_'+str(di_trial)):
                for i in file:
                    if 'weights' and 'feature' in i:
                        features = joblib.load(root+'/'+i)
                        labels = joblib.load(root+'/'+'labels')
            
            # features['train'][0] are genuine references, features['train'][1] are query signatures
            data = np.concatenate((features['train'][0], features['train'][1]), axis=1)
            label = labels['train']
            traindata_pos = data[labels['train']['labels']==1]
            traindata_neg = data[labels['train']['labels']==0]
            train_g_len=len(traindata_pos)
            train_f_len=len(traindata_neg)
            trainlabel_p=np.ones([train_g_len])
            trainlabel_n=np.zeros([train_f_len])
            now_batch_p=0
            now_batch_n=0
            batch_size_p=5
            batch_size_n=40
            total_batch_n=np.int(train_f_len/batch_size_n)+1
            num_of_pos=np.int(train_g_len/batch_size_p)
            num_of_neg=np.int(train_f_len/batch_size_n)
            train_p=traindata_pos
            train_n=traindata_neg
            for i in range(num_of_pos):
                now_batch_p=i
                if i < num_of_neg:
                    x_batch_train,y_batch_train = Functions.get_batch(train_p,trainlabel_p,train_n,trainlabel_n,batch_size_p,batch_size_n,now_batch_p,now_batch_n,total_batch_n)
                if i >= num_of_neg:
                    if now_batch_n % num_of_neg == 0:
                        now_batch_n = 0
                        train_n = traindata_neg
                    x_batch_train,y_batch_train = Functions.get_batch(train_p,trainlabel_p,train_n,trainlabel_n,batch_size_p,batch_size_n,now_batch_p,now_batch_n,total_batch_n)
                x_batch_train=np.float32(x_batch_train)
                y_batch_train=np.reshape(y_batch_train,[45,1])
                g, rate = sess.run([add_global, learning_rate])
                loss,losssummary, _ = sess.run([toprankloss,loss_summary,train_step], feed_dict={x: x_batch_train, y_ : y_batch_train, training:True})
                # train_writer.add_summary(losssummary, step)
                print("step: {}, train loss: {:g}".format(step, loss))
                now_batch_n=now_batch_n+1                
                step=step+1
            if (epoch % 1 == 0):
                print("Saving checkpoint")
                if not os.path.exists('/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/Top_rankNN/TopRankNN_models/'+str(di_dataset)+'/'+'CV'+str(di_cv)+'_'+str(di_trial)):
                    os.makedirs('/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/Top_rankNN/TopRankNN_models/'+str(di_dataset)+'/'+'CV'+str(di_cv)+'_'+str(di_trial))
                saver.save(sess, '/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/Top_rankNN/TopRankNN_models/'+str(di_dataset)+'/'+'CV'+str(di_cv)+'_'+str(di_trial)+'/p_'+str(num_p)+'/' + model_name + '_fc.ckpt')
            
            
if __name__ == '__main__':
    di_dataset, di_cv, di_trial = sys.argv[1], sys.argv[2], sys.argv[3]
    main(di_dataset, di_cv, di_trial)
