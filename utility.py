# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:30:16 2023

@author: Ghazal Azarfar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import date
from tabulate import tabulate
import tensorflow as tf
from scipy.optimize import curve_fit


def process(data):
    #This function removes extra variables and selects patients who have 12 months antobody information
    X = data['X'][:,1:40]#removing the hospital info
    X = np.delete(X, 23, axis = 1)#removing extra covid19 info

    Antibody = data['y_value']
    y = data['y']

    time_point = 5
    feature_size = X.shape[1]
    sample_count = int(X.shape[0]/time_point) 

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
    return X_12M,y_12M

def f(x, A, B): # this is a 'straight line' y=f(x). Predictions are finally fit to a straight line.
    return A*x + B

def Blant_Alman_plot(X,Y,mytitle,fold_number):
    #This function calculates the Pearson correlation coefficient, the equivalent mean square error and the Error<10% and Error <5%
    #This function also plots the Blant Alman Plot


    # Calculating the pearson correlation coefficient and storing it in PCC variable
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    PCC = (np.sum(np.multiply(X-X_mean,Y-Y_mean)))/np.sqrt(np.sum(np.square(X-X_mean))*np.sum(np.square(Y-Y_mean)))
    # Calculating the equivalent MST and storing it in mse variable
    mse = np.mean(np.square(Y-X))
    
    #calculating Error<10% and Error <5%
    distance = np.sqrt(np.power(X-Y,2))
    indx3 = np.where(distance > 0.65)
    distance_10_percent = 100-len(indx3[0])/len(distance)*100
    indx4 = np.where(distance > 0.65/2)
    distant_5_percent  = 100-len(indx4[0])/len(distance)*100
    

    #index0 stores indexes of responders that are correctly identified as responders
    indx0 = np.intersect1d(np.where(X.flatten()>np.log10(80)), np.where(Y.flatten()>np.log10(80)), assume_unique=True, return_indices=False)
    #index1 stores indexes of non-responders that are correctly identified as non-responders
    indx1 = np.intersect1d(np.where(X.flatten()<np.log10(80)), np.where(Y.flatten()<np.log10(80)), assume_unique=True, return_indices=False)
    #index2 stores indexes of cases that are not correctly identified
    indx2 = np.setdiff1d(np.array(range(0,len(X))), np.append(indx1, indx0), assume_unique=False)
    
    #plotting the Blat Alman plot
    x = np.linspace(-0.4,6.1,100)# range of x values
    #plotting the scatter dots
    plt.figure()    
    plt.scatter(X[indx0], Y[indx0], s=100, alpha=0.8, facecolors='#77AC30', edgecolors = '#77AC30')#green
    plt.scatter(X[indx1], Y[indx1], s=100, alpha=0.8, facecolors='#A2142F', edgecolors = '#A2142F')#red
    plt.scatter(X[indx2], Y[indx2], s=100, alpha=0.8, facecolors='#0072BD', edgecolors = '#0072BD')#blue
    #plotting the <10% Error lines
    plt.plot(x,x-0.65,color = '#000000', linestyle = 'dashed')
    plt.plot(x,x,color = '#000000')
    plt.plot(x,x+0.65,color = '#000000', linestyle = 'dashed')
    #plotting the Y=X line
    plt.plot(x,np.zeros((100,1))+np.log10(80),color = '#000000', linestyle = 'dashed')
    plt.plot(np.zeros((100,1))+np.log10(80),x,color = '#000000', linestyle = 'dashed')
    #title and labels for the plot
    plt.title(f'MSE = {mse:.2f} , Pearson_cor_coef =  {PCC:.2f}')
    plt.xlabel('log(Antibody)')
    plt.ylabel('predicted value')
    
    return distance_10_percent,distant_5_percent,mse,PCC

def my_cross_validate(X,y,model,cv,random_number,scale,myactivation,drop1,drop2,Learning_Rate,unit1,unit2,Epoch):  
    #defining variables
    Y_predicted_mean = np.zeros((cv,1))# mean of the predicted antibody values
    Y_predicted_var = np.zeros((cv,1))# variance of the predicted antibody values
    MSE = np.zeros((cv,1))# mean square error between the predicted antibody values and real antibody values 
    EqMSE = np.zeros((cv,1))#Equivalent mean square error between the predicted antibody values and real antibody values 
    PCC = np.zeros((cv,1))#Pearson corrolation coefficient
    distant_10_percent = np.zeros((cv,1))# <10% error
    distant_5_percent = np.zeros((cv,1))# <10% error
    train_time = np.zeros((cv,1))# trainig time
    n = int(np.floor(X.shape[0]/cv))#number of samples in each cross validation fold
    myaccuracy = np.zeros((1,1))
    for i in range(0,cv):
        val_range = range(n*i,n*(i+1)) #range of indexes used in each cross validation fold
        X_validation = X[val_range,:]#data of validation set
        y_validation = np.ravel(y[val_range])#labels of validation set
        X_train =  np.delete(X, val_range, axis = 0)#data of training set
        y_train =  np.ravel(np.delete(y, val_range, axis = 0))#labels of training set
        
        #Training
        start_time = time.time()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = Learning_Rate),loss=tf.keras.losses.MeanAbsoluteError(),metrics=['mse']) 
        history = model.fit(X_train, y_train, batch_size=64, epochs=Epoch, validation_split = 0.0, verbose = 0)
        myaccuracy = np.concatenate((myaccuracy,np.reshape(np.array(history.history['mse']),(Epoch,1))), axis = 0)

        plt.figure()
        plt.plot(myaccuracy)
        plt.title('model mse')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        train_time[i,0] = time.time() - start_time  
    
        
        #Validation
        y_predict = model.predict(X_validation,verbose=0)
        Y_predicted_mean[i,0] = np.mean(y_predict)#mean of the prediction
        Y_predicted_var[i,0] = np.var(y_predict)#variance of the predictio
        MSE[i,0] = np.mean(np.square(y_predict-y_validation))#mean square error of predictions
        
        #Finding the bias line
        y_train_per =  model.predict(X_train,verbose=0)    
        y_train_per = np.ravel((y_train_per - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376))
        y_predict = np.ravel((y_predict - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376))
        val_y = np.ravel((y_validation - 0.1)/0.8*5.87732526443497)
        train_y = np.ravel((y_train - 0.1)/0.8*5.87732526443497)
        popt, pcov = curve_fit(f,  train_y,  train_y - y_train_per )
        #plotting the Blant Alman plot after bias correction
        distant_10_percent[i,0],distant_5_percent[i,0],EqMSE[i,0],PCC[i,0] = Blant_Alman_plot(val_y,y_predict + f(y_predict, popt[0],popt[1]),'log(Antibodty)',i)


    #Creating a table to report the outputs     
    print(tabulate([[#random_number,
                     #scale,
                     #myactivation,
                     #drop1,
                     #drop2,
                     #dyna,
                     #Learning_Rate,
                     Epoch,
                     unit1,
                     unit2,
                     f'{np.mean(train_time):.2f} +- {np.std(train_time):.2f}',
                     f'{np.mean(MSE):.2f} +- {np.std(MSE):.2f}',
                     f'{np.mean(EqMSE):.2f} +- {np.std(EqMSE):.2f}',
                     f'{np.mean(Y_predicted_mean):.2f} +- {np.std(Y_predicted_mean):.2f}',
                     f'{ np.mean(Y_predicted_var):.2f} +- {np.std(Y_predicted_var):.2f}',
                     f'{np.mean(PCC):.2f} +- {np.std(PCC):.2f}',
                     f'{np.mean(distant_10_percent):.2f} +- {np.std(distant_10_percent):.2f}',
                     f'{np.mean(distant_5_percent):.2f} +- {np.std(distant_5_percent)  :.2f}'
                     ]], 
                    headers = [#'Ran Num',# random number
                               #'T', # Temperature
                               #'Activation',
                               'DO 1',#drop out layer 1
                               'DO 2',
                               #'dyna',
                               #'learn rate',
                               'epoch',
                               #'unit1',
                               #'unit2',
                               'Training time', # train time
                               'MSE',
                               'Eq MSE',
                               'Mean',
                               'variance',
                               'PCC', # pearson correlation coefficient
                               'Dist < 10%',
                               'Dist <5%'

                               ],tablefmt='orgtbl'))


    
    return model




def normalize(data):
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min)*0.8/(data_max-data_min)+0.1
    data[np.isnan(data)] = np.mean(data)
    return data

def normalize_categorized(data):
    categories =  data.unique()
    data = data.replace('MyNaN',0) 
    categories = np.delete(categories,np.where(categories == 'MyNaN'))
    data = data.replace('No',1) 
    categories = np.delete(categories,np.where(categories == 'No'))
    new_categories = np.arange(2, len(categories)+2, 1, dtype=float)
    n = 0
    for cat in categories:
        data = data.replace(cat, new_categories[n]) 
        n = n + 1
    return data.values



def data_stat(data, step, xlabel):
    filled_percentage = len(data[data.notna()])/len(data)*100
    # filled_percentage =0
    data = data[data.notna()]
    data = data.values
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_var = np.var(data)
    data_median = np.median(data)
    mybin = range(int(np.floor(data_min)),int(np.floor(data_max))+2, step)
    #plt.rcParams["figure.figsize"] = [7.00, 3.50]
    #plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.hist(data,mybin)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xlim((data_min-1, data_max+2)) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('min = {:.0f}'.format(data_min) + ', max = {:.0f}'.format(data_max) +', mean = {:.2f}'.format(data_mean) + ', median = {:.2f}'.format(data_median) + ', var = {:.2f}'.format(data_var) + ', filled = {:.0f}'.format(filled_percentage) + '%' ,fontsize=20)
    #manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle()
    plt.show()
    #plt.savefig( xlabel +'.png')
    return data_min, data_max, data_mean, data_var,filled_percentage


def data_pichart(data, mytitle):
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    mylabels =  data.unique()
    y = data.value_counts()
    percentage = y[mylabels]/np.sum(y[mylabels])*100
    plt.figure()
    plt.title(mytitle +', filled = {:.0f}'.format(filled_percentage) + '%')
    plt.pie(y[mylabels], labels = mylabels)
    plt.show() 
    if mytitle == 'Re-transplant?': 
        mytitle = 'Re-transplant'
    elif mytitle == 'Does the patient have chronic kidney disease (defined as eGFR < 30)':
        mytitle = 'Does the patient have chronic kidney disease'
    elif mytitle == 'Did the patient receive a different vaccine for the second dose?':
        mytitle ='Did the patient receive a different vaccine for the second dose'
    elif mytitle == 'Was pre-vaccine (serum) sample obtained?':
        mytitle ='Was pre-vaccine (serum) sample obtained'       
    # plt.savefig(mytitle +'.png')
    print(percentage)
    print(y)
    return 0


def days_to_month_year(data, xlabel):
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_var = np.var(data)
    data = data.values
    months_year = len(data[np.where((data <= 135))])
    months_year = np.reshape(months_year,(1,1))
    months_year.shape
    #axis_title = ['<3 month','3 month','6 month','9 month','1 year', ]
    #print('[less than ' + str(135) + ' days, or ' + str(135/30) + ' months]')
    for i in range (1,4):
        tmp =  len(data[np.where((data > 90*i+45) & (data <= 90*(i+1)+45))])
        months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(90*i+45) + ' to ' + str(90*(i+1)+45) + ']  ' + '[3 month interval is: ' + str((90*i+45)/90) + ' to ' + str((90*(i+1)+45)/90) + ']' )

    tmp = len(data[np.where((data > 90*(3+1)+45) & (data <= 182*(2+1)+90))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
    #print('[day interval is: ' + str(90*(3+1)+45) + ' to ' + str(182*(2+1)+90) + ']  ' + '[6 month interval is: ' + str((90*(3+1)+45)/182) + ' to ' + str((182*(2+1)+90)/182) + ']' )
    for i in range(3,10):
        tmp =  len(data[np.where((data > 182*i+90) & (data <= 182*(i+1)+90))])
        months_year = np.append(months_year,  np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(182*i+90) + ' to ' + str(182*(i+1)+90) + ']  ' + '[6 month interval is: ' + str((182*i+90)/182) + ' to ' + str((182*(i+1)+90)/182) + ']' )
    
    tmp =  len(data[np.where((data > 182*(9+1)+90) & (data <= 365*6))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0) 
    #print('[day interval is: ' + str(182*(9+1)+90) + ' to ' + str(365*6) + ']  ' + '[1 year interval is: ' + str((182*(9+1)+90)/365) + ' to ' + str(365*6/365) + ']' )
    
    for i in range(6,10):
        tmp =  len(data[np.where((data > 365*i) & (data <= 365*(i+1)))])
        months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(365*i) + ' to ' + str(365*(i+1)) + ']  ' + '[1 year interval is: ' + str((365*i)/365) + ' to ' + str((365*(i+1))/365) + ']' )
    tmp =  len(data[np.where((data > 365*(9+1)) & (data <= np.max(data)))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)

    axis_legends = ['< 3 month', '6 months', '9 months', '1 year', '1.5 years', '2 years', '2.5 years', '3 years', '3.5 years', '4 years', '4.5 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years', '> 10 years' ]
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.bar(axis_legends,months_year.flatten())
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('min = {:.0f}'.format(data_min) + ', max = {:.0f}'.format(data_max) +', mean = {:.2f}'.format(data_mean) + ', var = {:.2f}'.format(data_var) + ', filled = {:.0f}'.format(filled_percentage) + '%', fontsize=20)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.savefig( xlabel +'.png')
    return axis_legends, months_year


def select_patients_SOD(data, SOD):
    MyBolninx = SOD.isin(['Checked'])
    patients = data.to_dict('list')
    patients = pd.DataFrame(patients, index = MyBolninx)
    patients = patients.loc[True];
    patients= patients.set_index(np.array(range(0,len(patients))))
    return patients

def days_between_dates(df, header_date1, header_date2, day_column_name):
    # date_format = "%Y_%m_%d"
    
    date1 = df[header_date1[0],header_date1[1]]
    date2 = df[header_date2[0],header_date2[1]]
    df[day_column_name] = np.nan
    days = np.zeros((len(df),1))
    count = 0
    for row in df.loc[(df[header_date2[0],header_date2[1]].notna()) & (df[header_date1[0],header_date1[1]].notna())].index:
        # print(row)
        dt2 = str(date2[row])
        dt1 = str(date1[row])
        a = date(int(dt1[0:4]),int(dt1[5:7]),int(dt1[8:10]))
        b = date(int(dt2[0:4]),int(dt2[5:7]),int(dt2[8:10]))
        days[count] = (b-a).days
        count = count + 1

    return days, count


##############################################################################
# Columns selected for the analysis
##############################################################################
id_columns = [['Characteristics','Patient ID'],
              ['Characteristics','Data Access Group'],
              ['Characteristics','Age'],
              ['Characteristics','BMI'],
              ['Characteristics','Sex'],
              ['Characteristics','Organ Transplanted (check all that apply) (choice=Lung)'],
              ['Characteristics','Organ Transplanted (check all that apply) (choice=Heart)'],
              ['Characteristics','Organ Transplanted (check all that apply) (choice=Liver)'],
              ['Characteristics','Organ Transplanted (check all that apply) (choice=Kidney)'],
              ['Characteristics','Organ Transplanted (check all that apply) (choice=Pancreas)'],
              ['Characteristics','Re-transplant?'],
              ['Characteristics','Transplant Induction '],
              ['Characteristics','Drug of induction'],
              ['Characteristics','Treatment for rejection (in the past 3 months)'],
              ['Characteristics','Does the patient have chronic kidney disease (defined as eGFR < 30)'],
              ['Characteristics','Does the patient have any other immunosuppressive conditions (i.e. HIV, concurrent chemotherapy, etc)'],
              ['Characteristics','Transplant date'],
              ['Characteristics','Date of COVID-19 Vaccine Dose 1 date'],
              ['Characteristics','Vaccine dose 2 date'],
              ['Characteristics','Vaccine administered first dose'],
              ['Characteristics','Date of COVID-19 Vaccine Dose 1 date'],
              ['Characteristics','Vaccine dose 2 date'],
              ['Characteristics','Vaccine administered second dose'],
              ['Characteristics','Did the patient receive a different vaccine for the second dose?'],
              ['COVID Information', 'Did patient contract COVID-19 at any time during the study?'],
              ['COVID Information', 'post COVID Antibody results U/ml'],
              ['Visit 1 First Dose','Hospitalizations'],
              ['Visit 1 First Dose','Documented COVID-19 infection'],
              ['Visit 1 First Dose','Rejection since last visit\xa0'],
              ['Visit 1 First Dose','Prednisone'],
              ['Visit 1 First Dose','Cyclosporin'],
              ['Visit 1 First Dose','Tacrolimus'],
              ['Visit 1 First Dose','Sirolimus'],
              ['Visit 1 First Dose','Azathioprine'],
              ['Visit 1 First Dose','Mycophenolate mofetil or mycophenolate sodium'],
              ['Visit 1 First Dose','Prednisone Dose (mg)'],
              ['Visit 1 First Dose','Tacrolimus Dose (mg)'],
              ['Visit 1 First Dose','Most recent Tac level'],
              ['Visit 1 First Dose','Mycophenolate mofetil or mycophenolate sodium Dose (mg)'],
              ['Visit 1 First Dose','Prednisone Frequency'],
              ['Visit 1 First Dose','Tacrolimus Frequency'],
              ['Visit 1 First Dose','Mycophenolate mofetil or mycophenolate sodium Frequency'],
              ['Visit 2 Second Dose', 'Hospitalizations'],
              ['Visit 2 Second Dose', 'Documented COVID-19 infection'],
              ['Visit 2 Second Dose', 'Rejection since last visit\xa0'],
              ['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?'],
              ['Visit 2 Second Dose','Prednisone'],
              ['Visit 2 Second Dose','Cyclosporin'],
              ['Visit 2 Second Dose','Tacrolimus'],
              ['Visit 2 Second Dose','Sirolimus'],
              ['Visit 2 Second Dose','Azathioprine'],
              ['Visit 2 Second Dose','Mycophenolate mofetil or mycophenolate sodium'],
              ['Visit 2 Second Dose','Prednisone Dose (mg)'],
              ['Visit 2 Second Dose','Tacrolimus Dose (mg)'],
              ['Visit 2 Second Dose','Most recent Tac level'],
              ['Visit 2 Second Dose','Mycophenolate mofetil or mycophenolate sodium Dose (mg)'],
              ['Visit 2 Second Dose','Prednisone Frequency'],
              ['Visit 2 Second Dose','Tacrolimus Frequency'],
              ['Visit 2 Second Dose','Mycophenolate mofetil or mycophenolate sodium Frequency'],
              ['Third Dose Information','Date of third dose\xa0'],
              ['Third Dose Information','Vaccine administered third dose'],
              ['Third Dose Information', 'Did the patient receive a different vaccine for the third dose?'],
              ['Third Dose Information', 'Any changes in immunosuppression medication since last visit?'],
              ['Third Dose Information','Prednisone'],
              ['Third Dose Information','Cyclosporin'],
              ['Third Dose Information','Tacrolimus'],
              ['Third Dose Information','Sirolimus'],
              ['Third Dose Information','Azathioprine'],
              ['Third Dose Information','Mycophenolate mofetil or mycophenolate sodium'],
              ['Third Dose Information','Prednisone Dose (mg)'],
              ['Third Dose Information','Tacrolimus Dose (mg)'],
              ['Third Dose Information','Most recent Tac level'],
              ['Third Dose Information','Mycophenolate mofetil or mycophenolate sodium Dose (mg)'],
              ['Third Dose Information','Prednisone Frequency'],
              ['Third Dose Information','Tacrolimus Frequency'],
              ['Third Dose Information','Mycophenolate mofetil or mycophenolate sodium Frequency'],
              ['6M Visit' , '6 month follow-up sample collection date'],
              ['6M Visit','Hospitalizations'],
              ['6M Visit','Documented COVID-19 infection'],
              ['6M Visit','Rejection since last visit\xa0'],
              ['6M Visit', 'Any changes in immunosuppression medication since last visit?'],
              ['6M Visit','Prednisone'],
              ['6M Visit','Cyclosporin'],
              ['6M Visit','Tacrolimus'],
              ['6M Visit','Sirolimus'],
              ['6M Visit','Azathioprine'],
              ['6M Visit','Mycophenolate mofetil or mycophenolate sodium'],
              ['6M Visit','Prednisone Dose (mg)'],
              ['6M Visit','Tacrolimus Dose (mg)'],
              ['6M Visit','Most recent Tac level'],
              ['6M Visit','Mycophenolate mofetil or mycophenolate sodium Dose (mg)'],
              ['6M Visit','Prednisone Frequency'],
              ['6M Visit','Tacrolimus Frequency'],
              ['6M Visit','Mycophenolate mofetil or mycophenolate sodium Frequency'],
              ['6M Visit' , '6 month follow-up sample collection date'],
              ['12M Visit','12 month follow-up sample collection date'],
              ['12M Visit','Hospitalizations'],
              ['12M Visit','Documented COVID-19 infection'],
              ['12M Visit','Rejection since last visit\xa0'],
              ['12M Visit', 'Any changes in immunosuppression medication since last visit?'],
              ['12M Visit','Prednisone'],
              ['12M Visit','Cyclosporin'],
              ['12M Visit','Tacrolimus'],
              ['12M Visit','Sirolimus'],
              ['12M Visit','Azathioprine'],
              ['12M Visit','Mycophenolate mofetil or mycophenolate sodium'],
              ['12M Visit','Prednisone Dose (mg)'],
              ['12M Visit','Tacrolimus Dose (mg)'],
              ['12M Visit','Most recent Tac level\xa0'],
              ['12M Visit','Mycophenolate mofetil or mycophenolate sodium Dose (mg)'],
              ['12M Visit','Prednisone Frequency'],
              ['12M Visit','Tacrolimus Frequency'],
              ['12M Visit','Mycophenolate mofetil or mycophenolate sodium Frequency'],
              ['Fourth Dose','Date of fourth dose\xa0'],
              ['Antibody Information', 'Pre-vaccine Antibody results (U/ml)'],
              ['Fourth Dose', 'Did the patient receive a fourth dose?']]