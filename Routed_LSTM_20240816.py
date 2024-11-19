# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:46:58 2024

@author:Ghazal Azarfar
"""
import numpy as np
import random as rn
from utility import my_cross_validate,process
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Rescaling,Masking,Layer
import tensorflow as tf  







class MyExpand(Layer):
  def __init__(self):
    super(MyExpand, self).__init__()
    self.supports_masking = True
  def call(self, x):
    return tf.expand_dims(x,axis = -3)


class MyLayer(Layer):
  def __init__(self,unit1):
    super(MyLayer, self).__init__()
    self.variable = tf.Variable(tf.random_normal_initializer()(shape=[1,5,unit1]), dtype=tf.float32, name="PoseEstimation", trainable=True)
    self.supports_masking = True
  def call(self, inputs):
    return tf.multiply(inputs,self.variable)

class MyReduceSum(Layer):
  def __init__(self):
    super(MyReduceSum, self).__init__()
    self.supports_masking = True
  def call(self, u_hat,b):
    c = tf.nn.softmax(b,axis = 1)
    return tf.reduce_sum(tf.matmul(c,u_hat), axis = 1, keepdims = True)

class squash(Layer):
  def __init__(self,epsilon = 1e-7):
    super(squash, self).__init__()
    self.epsilon = epsilon
    self.supports_masking = True
  def call(self, s):
    s_norm = tf.norm(s, axis=-1, keepdims=True)
    return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + self.epsilon)

class Agreed(Layer):
  def __init__(self):
    super(Agreed, self).__init__()
    self.supports_masking = True
  def call(self, u_hat,v):
    return  tf.matmul(u_hat, v, transpose_b=True)

class MySqueeze(Layer):
  def __init__(self):
    super(MySqueeze, self).__init__()
    self.supports_masking = True
  def call(self,v):
    return tf.squeeze(v,axis = 1)


def Routed_LSTM_Covid(input_shape,scale,myactivation,drop1,drop2,unit1,unit2,dyna):
    
    #model = Sequential()
    #model.add()
    inputs = Input(shape=input_shape) 
    x = Masking(mask_value= 0,input_shape=input_shape)(inputs)
    # x = Embedding(input_dim = 10, output_dim = 5)(inputs)
    x = LSTM(unit1, activation='tanh', recurrent_activation='tanh', return_sequences=True)(x)#5,8
    x = Dropout(drop1)(x)
    u = MyExpand()(x) # u.shape: (None, 509, 1, 8)
    u_hat = MyLayer(unit1)(u)
    b = tf.zeros((input_shape[0],5,5))
    for i in range(dyna):
        #c = tf.nn.softmax(b,axis = 1)
        #s = tf.reduce_sum(tf.matmul(c,u_hat), axis = 1, keepdims = True)#5,8
        s = MyReduceSum()(u_hat,b)
        v = squash()(s)#5,8
        agreement = Agreed()(u_hat, v)
        b += agreement
        v = MySqueeze()(v)
    x = LSTM(unit2, activation='tanh', recurrent_activation='tanh')(v)
    x = Dropout(drop2)(x)
    x = Rescaling(scale, offset=0.0)(x)
    outputs = Dense(1, activation=myactivation)(x)
    model = Model(inputs, outputs)

    return model



path2data = 'D:\\UHN\\Covid19 Vaccination\\SOT_COVID_Data_Long_20240102.npz'
data = np.load(path2data)    
X_12M, y_12M = process(data)
time_point = X_12M.shape[1]
feature = X_12M.shape[2]

input_shape = ( time_point, feature)

cros_Val_num = 5


for myseed in [100]:#,2024,800,500,705,2023,2005,208,803,42,36]:
    for scale in [2]:#[0.5,1,1.5,2,2.5,3,4]: 
        for myactivation in ['swish']:#['tanh','softmax','relu','hard_sigmoid','elu','swish','softplus']:
            for drop1 in [0.2]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for drop2 in [0.2]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    for Learning_rate in [0.005]:#[0.1,0.01,0.005,0.001,0.0005,0.0001]:
                        for unit1 in [8,16,32,64,128,256]:
                            for unit2 in [8,16,32,64,128,256]:
                                for epoch in [100]:#range(50,500,25):
                                    for dyna in [3]:#[1,2,3]:
                                        np.random.seed(myseed)
                                        tf.random.set_seed(myseed)
                                        rn.seed(myseed)
                                        tf.experimental.numpy.random.seed(myseed)
                                        model = Routed_LSTM_Covid(input_shape,scale,myactivation,drop1,drop2,unit1,unit2,dyna)
                                        model = my_cross_validate(X_12M,y_12M,model,cros_Val_num,myseed,scale,myactivation,drop1,drop2,Learning_rate,unit1,unit2,epoch)
                
    


#path2save = 'D:\\UHN\\Covid19 Vaccination\\LSTM\\12 Months\\regression\\20240322\\'


#model.save(path2save)

# from scipy.stats import ks_2samp
# for i in [0,2,29]:#range(0,X.shape[2]):
#     print('***************************************')
#     print(str(i))
#     print('***************************************')
#     print(ks_2samp(np.reshape(normal[:,:,i],(normal.shape[0]*5)), np.reshape(outliers[:,:,i],(outliers.shape[0]*5))))
    
    



