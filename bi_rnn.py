from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from function import *
from tensorflow.contrib import rnn

h=64
w=64
feature_size=w*h; channels=1
testing_step=100;
training_steps = 200000
# Network Parameters
number_of_layers=2; #Start from only one layer

def ConvLSTM(x):
    convlstm_layer1 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,channels],
                 output_channels=8,
                 kernel_shape=[4,4],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell1")
    convlstm_layer2 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,8],
                 output_channels=16,
                 kernel_shape=[8,8],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell2")
    convlstm_layer3 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,  
                 input_shape=[h,w,16],
                 output_channels=1,
                 kernel_shape=[8,8],
                 use_bias=True, 
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell3")

    convlstm_layer11 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,channels],
                 output_channels=8,
                 kernel_shape=[4,4],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell11")
    convlstm_layer12 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,8],
                 output_channels=16,
                 kernel_shape=[8,8],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell12")
    convlstm_layer13 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,     
                 input_shape=[h,w,16],
                 output_channels=1,  
                 kernel_shape=[8,8],
                 use_bias=True,    
                 skip_connection=False,
                 forget_bias=0.5,  
                 initializers=None, 
                 name="conv_lstm_cell13")
    lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
        [convlstm_layer1, convlstm_layer2, convlstm_layer3])
    lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
        [convlstm_layer11, convlstm_layer12, convlstm_layer13])

    print(lstm_fw_cell,lstm_bw_cell)
    initial_state=lstm_fw_cell.zero_state(FLAGS.batch_size, dtype=tf.float32 )
    #Things to Do
    outputs, states=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=x, sequence_length=None, initial_state_fw=initial_state,initial_state_bw=initial_state,dtype=tf.float32)

    return outputs, states
# Didn't apply drop_out here

