# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:46:58 2024

@author:Ghazal Azarfar
"""
import numpy as np
import tensorflow as tf  
import random as rn
from utility import my_cross_validate,process
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Rescaling,Masking



def LSTM_Covid(input_shape,scale,myactivation,drop1,drop2,unit1,unit2):
    
    inputs = Input(shape=input_shape) 
    x = Masking(mask_value= 0,input_shape=input_shape)(inputs)
    # x = Embedding(input_dim = 10, output_dim = 5)(inputs)
    x = LSTM(unit1, activation='tanh', recurrent_activation='tanh', return_sequences=True)(x)#5,8
    x = Dropout(drop1)(x)
    x = LSTM(unit2, activation='tanh', recurrent_activation='tanh')(x)
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
            for drop1 in [0.3]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for drop2 in [0.6]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    for Learning_rate in [0.005]:#[0.1,0.01,0.005,0.001,0.0005,0.0001]:
                        for unit1 in [16]:#[8,16,32,64,128,256]:
                            for unit2 in [64]:#[8,16,32,64,128,256]:
                                for epoch in [400]:#range(50,500,25):
                                        np.random.seed(myseed)
                                        tf.random.set_seed(myseed)
                                        rn.seed(myseed)
                                        tf.experimental.numpy.random.seed(myseed)
                                        model = LSTM_Covid(input_shape,scale,myactivation,drop1,drop2,unit1,unit2)
                                        model = my_cross_validate(X_12M,y_12M,model,cros_Val_num,myseed,scale,myactivation,drop1,drop2,Learning_rate,unit1,unit2,epoch)
                
    


#path2save = 'D:\\UHN\\Covid19 Vaccination\\LSTM\\12 Months\\regression\\20240322\\'


#model.save(path2save)

# from scipy.stats import ks_2samp
# for i in [0,2,29]:#range(0,X.shape[2]):
#     print('***************************************')
#     print(str(i))
#     print('***************************************')
#     print(ks_2samp(np.reshape(normal[:,:,i],(normal.shape[0]*5)), np.reshape(outliers[:,:,i],(outliers.shape[0]*5))))
    
    
