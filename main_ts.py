from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from rnn import *
import numpy as np
import skimage.measure
parser = argparse.ArgumentParser()


#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.005 #0.001
X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels]) 
Y = tf.placeholder("float", [FLAGS.batch_size, None, h,w,3])
timesteps = tf.shape(X)[1]
h=tf.shape(X)[2] #h:256
w=tf.shape(X)[3] #w:513

prediction, last_state = ConvLSTM(X) 
loss_op=tf.losses.mean_pairwise_squared_error(Y,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    test("temp",sess,loss_op,train_op,X,Y,prediction,last_state,fout_log)
    start=[0,20,40,60,80]
    end=[20,40,60,80,102]

    for ii in range(1000):
        name=str(ii)
        for k in range(len(start)):
            train_X,train_Y,val_X,val_Y=read_input("./data_ts3/",start[ii],end[ii])
            train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log)
            test(name,sess,loss_op,train_op,X,Y,prediction,last_state,fout_log)

        test2(name,sess,loss_op,train_op,X,Y,val_X,val_Y,prediction,last_state,fout_log)
        if ii%5 == 0 : 
            test(name,sess,loss_op,train_op,X,Y,prediction,last_state,fout_log)
        save_path = saver.save(sess, "./model_"+str(ii)+".ckpt")
fout_log.close();

