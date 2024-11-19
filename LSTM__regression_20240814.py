

import numpy as np
import tensorflow as tf  
import random as rn

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
# permutation_importance

import time


# sess = tf.compat.v1.keras.backend.get_session()
# tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input


# print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B


def Blant_Alman_plot(X,Y,mytitle,figurenumber):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    PCC = (np.sum(np.multiply(X-X_mean,Y-Y_mean)))/np.sqrt(np.sum(np.square(X-X_mean))*np.sum(np.square(Y-Y_mean)))
    mse = np.mean(np.square(Y-X))
    x = np.linspace(-0.4,6.1,100)
    plt.figure(figurenumber)
    
    indx0 = np.intersect1d(np.where(X.flatten()>np.log10(80)), np.where(Y.flatten()>np.log10(80)), assume_unique=True, return_indices=False)
    plt.scatter(X[indx0], Y[indx0], s=100, alpha=0.8, facecolors='#77AC30', edgecolors = '#77AC30')#green
    
    indx1 = np.intersect1d(np.where(X.flatten()<np.log10(80)), np.where(Y.flatten()<np.log10(80)), assume_unique=True, return_indices=False)
    plt.scatter(X[indx1], Y[indx1], s=100, alpha=0.8, facecolors='#A2142F', edgecolors = '#A2142F')#red
    
    indx2 = np.setdiff1d(np.array(range(0,len(X))), np.append(indx1, indx0), assume_unique=False)
    plt.scatter(X[indx2], Y[indx2], s=100, alpha=0.8, facecolors='#0072BD', edgecolors = '#0072BD')#blue
    
    distance = np.sqrt(np.power(X-Y,2))
    
    indx3 = np.where(distance > 0.65)
    indx4 = np.where(distance > 0.65/2)
    # plt.scatter(X[indx3], Y[indx3], s=100, alpha=0.8, facecolors='#FFC0CB', edgecolors = '#FFC0CB')
    
    # indx4 = np.setdiff1d(np.where(X.flatten() < X.flatten()-0.65), np.where(Y.flatten() < X.flatten()-0.65), assume_unique=False)
    # plt.scatter(X[indx4], Y[indx4], s=100, alpha=0.8, facecolors='#800080', edgecolors = '#800080')
    
    plt.plot(x,x-0.65,color = '#000000', linestyle = 'dashed')
    plt.plot(x,x,color = '#000000')
    plt.plot(x,x+0.65,color = '#000000', linestyle = 'dashed')
    plt.plot(x,np.zeros((100,1))+np.log10(80),color = '#000000', linestyle = 'dashed')
    plt.plot(np.zeros((100,1))+np.log10(80),x,color = '#000000', linestyle = 'dashed')
    plt.title(f'MSE = {mse:.2f} , Pearson_cor_coef =  {PCC:.2f}')
    print(f'MSE = {mse:.2f} , Pearson_cor_coef =  {PCC:.2f}')
    plt.xlabel('log(Antibody)')
    plt.ylabel('predicted value')
    return indx0,indx1,indx2,100-len(indx3[0])/len(distance)*100,100-len(indx4[0])/len(distance)*100,mse,PCC


def squash(s, epsilon = 1e-7):
  s_norm = tf.norm(s, axis=-1, keepdims=True)
  return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)

class MyLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyLayer, self).__init__()

    #your variable goes here#[1,5,8]
    self.variable = tf.Variable(tf.random_normal_initializer()(shape=[1,5,8]), dtype=tf.float32, name="PoseEstimation", trainable=True)


  def call(self, inputs):

    # your mul operation goes here
    x = tf.multiply(inputs,self.variable)
    
    return x

def LSTM_Covid(input_shape):
    
    #model = Sequential()
    #model.add()
    inputs = Input(shape=input_shape) 
    x = tf.keras.layers.Masking(mask_value= 0,input_shape=input_shape)(inputs)
    # x = Embedding(input_dim = 10, output_dim = 5)(inputs)
    x = LSTM(8, activation='tanh', recurrent_activation='tanh', return_sequences=True)(x)#5,8
    x = Dropout(0.2)(x)
    x = LSTM(8, activation='tanh', recurrent_activation='tanh')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='tanh')(x)
    model = Model(inputs, outputs)

    return model

def my_cross_validate(X,y,model,cros_Val_num):  
    def weighted_mean_absolute_error(y_true, y_pred):
        class_weights = {}
        for i in range(0,len(y_true)):
            if y_true[i] < 0.137:
                class_weights[str(y_true[i])] = 0.152
            elif (y_true[i] >= 0.137) & (y_true[i] < 0.39):
                class_weights[str(y_true[i])] = 0.759
            elif (y_true[i] >= 0.39) & (y_true[i] < 0.63):
                class_weights[str(y_true[i])] = 0.759/3
            else:
                class_weights[str(y_true[i])] = 0.088
        loss =  tf.math.reduce_mean(tf.math.multiply(tf.math.square(tf.cast(y_true, tf.float32) - y_pred), list(class_weights.values())))

        return loss

    loss = np.zeros((cros_Val_num,1))
    mae = np.zeros((cros_Val_num,1))
    acc = np.zeros((cros_Val_num,1))
    indx0 = {}
    indx1 = {}
    indx2 = {}
    n_fold = int(np.floor(X.shape[0]/cros_Val_num))
    mse = np.zeros((5,1))
    PCC = np.zeros((5,1))
    indx3 = np.zeros((5,1))
    indx4 = np.zeros((5,1))
    
    indx00 = {}
    indx10 = {}
    indx20 = {}
    mse0 = np.zeros((5,1))
    PCC0 = np.zeros((5,1))
    indx30 = np.zeros((5,1))
    indx40 = np.zeros((5,1))
    for i in range(0,cros_Val_num):#cros_Val_num
        myaccuracy = []
        

        val_range = range(n_fold*i,n_fold*(i+1)) 


        X_val = X[val_range,:]
        y_val = np.ravel(y[val_range])


        X_train =  np.delete(X, val_range, axis = 0)
        y_train =  np.ravel(np.delete(y, val_range, axis = 0))
        
        
        indx = np.where(~X_val[:,4,33:X.shape[2]].any(axis=1))[0]
        tmp = y_val[indx]
        y_val = y_val[~np.all(X_val[:,4,33:X.shape[2]] == 0, axis=1)]
        y_val = np.concatenate((y_val,tmp),axis = 0)
        tmp = X_val[indx,:,:]
        X_val = X_val[~np.all(X_val[:,4,33:X.shape[2]] == 0, axis=1),:,:]
        X_val = np.concatenate((X_val,tmp),axis = 0)

        
        indx = np.where(~X_train[:,4,33:X.shape[2]].any(axis=1))[0]
        tmp = y_train[indx]
        y_train = y_train[~np.all(X_train[:,4,33:X.shape[2]] == 0, axis=1)]
        y_train = np.concatenate((y_train,tmp),axis = 0)
        tmp = X_train[indx,:,:]
        X_train = X_train[~np.all(X_train[:,4,33:X.shape[2]] == 0, axis=1),:,:]
        X_train = np.concatenate((X_train,tmp),axis = 0)

        # model = LSTM_Covid((5,54))
        start_time = time.time()
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),loss=tf.keras.losses.MeanSquaredError(),metrics=['mse']) 
        history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split = 0, verbose = 0)
    

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),loss=tf.keras.losses.MeanSquaredError(),metrics=['mse']) 
        history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split = 0, verbose = 0)


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),loss=tf.keras.losses.MeanSquaredError(),metrics=['mse']) 
        history = model.fit(X_train, y_train, batch_size=64, epochs=200, validation_split = 0, verbose = 0)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        myaccuracy.append(history.history['mse'])
        # myval_accuracy.append(history.history['val_mae'])
        
        
        myaccuracy = list(np.concatenate(myaccuracy).flat)
        # myval_accuracy = list(np.concatenate(myval_accuracy).flat)
        
        plt.figure(2*i)
        plt.plot(myaccuracy)
        # plt.plot(myval_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        
        loss[i], mae[i] = model.evaluate(X_val, y_val, batch_size=1)
        
        y_predict = model.predict(X_val)
        y_train_per =  model.predict(X_train)     
        y_train_per = np.ravel((y_train_per - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376))
        y_predict = np.ravel((y_predict - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376))
        val_y = np.ravel((y_val - 0.1)/0.8*5.87732526443497)
        train_y = np.ravel((y_train - 0.1)/0.8*5.87732526443497)
        
        popt, pcov = curve_fit(f,  train_y,  train_y - y_train_per )
        
        # indx00[str(i)],indx10[str(i)],indx20[str(i)],indx30[i,0],indx40[i,0],mse0[i,0],PCC0[i,0] = Blant_Alman_plot(train_y,y_train_per + f(y_train_per, popt[0],popt[1]),'log(Antibodty)',2*i+1)
        indx0[str(i)],indx1[str(i)],indx2[str(i)],indx3[i,0],indx4[i,0],mse[i,0],PCC[i,0] = Blant_Alman_plot(val_y,y_predict + f(y_predict, popt[0],popt[1]),'log(Antibodty)',2*i+1)

            # 
        y_predict[y_predict < np.log10(80)] = 0
        y_predict[y_predict >= np.log10(80)] = 1
        
        val_y[val_y < np.log10(80)] = 0
        val_y[val_y >= np.log10(80)] = 1
        
        y_train_per[y_train_per < np.log10(80)] = 0
        y_train_per[y_train_per >= np.log10(80)] = 1
        
        train_y[train_y < np.log10(80)] = 0
        train_y[train_y >= np.log10(80)] = 1
        # print('***************************************************')
        # print('Classification Report - train')
        # print('***************************************************')
        # print(classification_report(train_y, y_train_per, labels = [0,1], target_names=['not Immune','Immune']))
        # # cm = confusion_matrix(np.reshape(y_val,(sample_count*time_point)), np.reshape(y_predict,(sample_count*time_point)))
        # print('***************************************************')
    
        print('***************************************************')
        print('Classification Report - test')
        print('***************************************************')
        print(classification_report(val_y, y_predict, labels = [0,1], target_names=['not Immune','Immune']))
        # cm = confusion_matrix(np.reshape(y_val,(sample_count*time_point)), np.reshape(y_predict,(sample_count*time_point)))
        print('***************************************************')
    tmp0 = np.mean(mse0)
    tmp1 = np.std(mse0)
    
    tmp3 = np.mean(PCC0)
    tmp4= np.std(PCC0)

    tmp5 = np.mean(indx30)
    tmp6 = np.std(indx30)


    tmp7 = np.mean(indx40)
    tmp8 = np.std(indx40)
    print('***************************************************')
    print('cross-validation results - training')
    print(f'MSE = {tmp0:.2f} +- {tmp1:.2f} , Pearson_cor_coef =  {tmp3:.2f} +- {tmp4:.2f}')
    print(f'dist_more_65 =  {tmp5:.2f} +- {tmp6:.2f}, dist_more_32.5 =  {tmp7:.2f} +- {tmp8:.2f},')


    tmp0 = np.mean(mse)
    tmp1 = np.std(mse)
    
    tmp3 = np.mean(PCC)
    tmp4= np.std(PCC)

    tmp5 = np.mean(indx3)
    tmp6 = np.std(indx3)


    tmp7 = np.mean(indx4)
    tmp8 = np.std(indx4)
    print('***************************************************')
    print('cross-validation results - test')
    print(f'MSE = {tmp0:.2f} +- {tmp1:.2f} , Pearson_cor_coef =  {tmp3:.2f} +- {tmp4:.2f}')
    print(f'dist_more_65 =  {tmp5:.2f} +- {tmp6:.2f}, dist_more_32.5 =  {tmp7:.2f} +- {tmp8:.2f},')

    
    return loss, mae, acc, model,indx0,indx1,indx2,indx3,indx4,mse,PCC

path2data = 'D:\\UHN\\Covid19 Vaccination\\'
#fname = 'SOT_COVID_Data_Long_6M_padded_event_days.npz'
fname = 'SOT_COVID_Data_Long_20231027.npz'

data = np.load(path2data + fname)    



X = data['X']
Antibody = data['y_value']
y = data['y']

time_point = 5
feature_size = X.shape[1]
sample_count = int(X.shape[0]/time_point) 

epoch = 20
pc_num = 20


# myseed = 5000
# np.random.seed(myseed)
# tf.random.set_seed(myseed)
# rn.seed(myseed)
# X,pc_percentage,pca = data_reduc(X, pc_num)

feature = feature_size

X = np.reshape(X,[sample_count,time_point,feature])
Antibody = np.reshape(Antibody,[sample_count,time_point])
y = np.reshape(y,[sample_count,time_point])


p = np.random.default_rng(seed=66).permutation(sample_count)
X = X[p,:,:]
Antibody = Antibody[p,:]
y = y[p,:]


y_12M = np.zeros((X.shape[0],1))
X_12M = np.zeros(np.shape(X))
y_12M_class = np.zeros((X.shape[0],1))


#6M
n = 0
j = 0
for i in range(0,len(Antibody)):
    if np.isinf(Antibody[i,4]):
        j = j + 1
        # if np.isinf(Antibody[i,3]):
        #     j = j + 1
        # else: 
        #     y_12M[n,0] = Antibody[i,3]
        #     X_12M[n,0:4,:] = X[i,0:4,:]
        #     n = n + 1
    else:
        y_12M[n,0] = Antibody[i,4]
        y_12M_class[n,0] = y[i,4]
        X_12M[n,:,:] = X[i,:,:]
        n = n + 1
        

y_12M_class = np.ravel(y_12M_class[0:n,0])
y_12M = np.ravel(y_12M[0:n,0])
X_12M = X_12M[0:n,:,:]

input_shape = ( time_point, feature)
#learning rate =0.01
#950(0.1254+0.027)
#705(0.1305+-0.285)
#42(0.1224+-0.014)
#802(0.1268+-0.02)
myseed = 100#62,99(0.76),100(0.78+0.05)
np.random.seed(myseed)
tf.random.set_seed(myseed)
rn.seed(myseed)
tf.experimental.numpy.random.seed(myseed)

model = LSTM_Covid(input_shape)
# model.summary()

cros_Val_num = 5





loss, mae, acc, model,indx0,indx1,indx2,indx3,indx4,mse,PCC = my_cross_validate(X_12M,y_12M,model,cros_Val_num)




y_predict = model.predict(X_12M)

y_predict = np.ravel((y_predict - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376))
val_y = np.ravel((y_12M - 0.1)/0.8*5.87732526443497)


print('***************************************************')
print('cross-validation results')
print(mean_absolute_error(val_y, y_predict))

indx00,indx11,indx22,indx3,indx44,mse_final,PCC = Blant_Alman_plot(y_predict,val_y,'log(Antibody)',12)



print('***************************************************')

path2save = 'D:\\UHN\\Covid19 Vaccination\\LSTM without PCA\\6_12 Months\\regression\\20231103\\'


#model.save(path2save)

# from scipy.stats import ks_2samp
# for i in [0,2,29]:#range(0,X.shape[2]):
#     print('***************************************')
#     print(str(i))
#     print('***************************************')
#     print(ks_2samp(np.reshape(normal[:,:,i],(normal.shape[0]*5)), np.reshape(outliers[:,:,i],(outliers.shape[0]*5))))
    
    