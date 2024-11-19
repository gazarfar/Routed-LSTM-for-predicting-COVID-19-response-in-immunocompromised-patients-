# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:59:56 2023

@author: azarf
"""

# %matplotlib qt 

import shap
import pandas as pd
import time
import numpy as np
# Modelling
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, auc,roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV,StratifiedKFold, cross_val_score

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

# feature importance
import matplotlib.pyplot as plt

from feature_names_wide import feature_names_wide
from feature_names_longformat import feature_names_longformat

from sklearn.decomposition import PCA


from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B





def data_reduc(X, pc_num):
    #This function cancluates the PC scores and returns the 
    #reduced data
    pca = PCA(n_components=pc_num)

    pca.fit(X)
    myper = pca.explained_variance_ratio_
    scores = pca.transform(X)

    X = scores
    return X, myper


def optimize_forest_nodes(X_train,y_train, param_dist, mycv):

#803 - 5.58
    # Create a random forest classifier
    regr = RandomForestRegressor(random_state = 803)

    # Use random search to find the best hyperparameters
    grid_search = GridSearchCV(estimator= regr, param_grid= param_dist,
                               cv=KFold(n_splits=mycv, shuffle=True, random_state=803))

    # Fit the random search object to the data
    grid_search.fit(X_train, y_train)


    df = pd.concat([pd.DataFrame(grid_search.cv_results_["rank_test_score"], columns=["rank_test_score"]),
               pd.DataFrame(grid_search.cv_results_["params"]),
               pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"]),
               pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
    best_rf = grid_search.best_estimator_

    #Print the best hyperparameters
    print('***************************************************')
    print('Best hyperparameters:',  grid_search.best_params_)
    print('***************************************************')
    print(df.sort_values("rank_test_score"))
    print('***************************************************')
    n_estimator = grid_search.best_params_['n_estimators']
    max_depth = grid_search.best_params_['max_depth']
    return best_rf, n_estimator, max_depth


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
    # print(f'MSE = {mse:.2f} , Pearson_cor_coef =  {PCC:.2f}')
    plt.xlabel('log(Antibody)')
    plt.ylabel('predicted value')
    return indx0,indx1,indx2,100-len(indx3[0])/len(distance)*100,100-len(indx4[0])/len(distance)*100,mse,PCC

def optimize_weights(weights,param_grid, rf,X_train,y_train):
    gridsearch = GridSearchCV(estimator= rf, param_grid= param_grid,
                          cv=KFold(n_splits=5, shuffle=True, random_state=1))
    
    weight_search = gridsearch.fit(X_train, y_train)
    plt.figure(figsize=(12,8))
    weigh_data = pd.DataFrame({ 'score': weight_search.cv_results_['mean_test_score'], 'weight': (1- weights)})
    plt.plot(weigh_data['weight'], weigh_data['score'])
    plt.xlabel('Weight for class 1')
    plt.ylabel('F1 score')
    plt.xticks([round(i/10,1) for i in range(0,11,1)])
    plt.title('Scoring for different class weights', fontsize=24)
    return weight_search, weigh_data






##############################################
# Read and preprocess data
##############################################



path2data = 'C:\\Ghazal\\Covid19 Vaccination\\'
#fname = 'SOT_COVID_Data_Long_6M_padded_event_days.npz'
fname = 'SOT_COVID_Data_Long_20240102.npz'
# fname = 'SOT_COVID_Data_Long_20231121.npz'

data = np.load(path2data + fname)    



X = data['X'][:,1:40]
y = data['y']
Antibody = data['y_value']

X = np.reshape(X,[303,5,39])
X = np.delete(X, 23, axis = 2)

Antibody = np.reshape(Antibody,[303,5])


y_antibody = np.zeros((len(y),1))
X_12M = np.zeros((303,38))
n = 0
j = 0
for i in range(0,len(Antibody)):
    if np.isinf(Antibody[i,4]):
        j = j + 1
    else:
        y_antibody[n] = Antibody[i,4]
        X_12M[n,:] = X[i,2,:]
        X_12M[n,19] = X[i,0,19]
        n = n + 1

y_antibody = y_antibody[0:n]
X_12M = X_12M[0:n,:]

p = np.random.default_rng(seed=807).permutation(n)
X_12M = X_12M[p,:]


y_antibody = y_antibody[p]



        


y_antibody = np.ravel(y_antibody)

data_X = X_12M

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_X, np.ravel(y_antibody), test_size=0.2, random_state = 20230901)


##############################################
# Model development
##############################################
my_class_weight = {0:  0.6263265306122449, 1: 0.37367346938775514}
# my_class_weight = {0:  0.06061224489795918, 1: 0.9393877551020409}

param_dist = {'n_estimators': [8,16,50,100,150,200,250],#randint(50,500),
               'max_depth':   [4,8,16,32]}
                 # 'class_weight': [{0:w, 1:1.0-w} for w in weights]}#randint(1,20)}
                 
mycv = 5                  
# best_rf, n_estimator, max_depth = optimize_forest_nodes(X_train,y_train, param_dist,mycv)
# 
param_grid = {
    'max_depth': [5,10,20],
    'max_features' : [5,10,20],
    'n_estimators': [20,50]}


start_time = time.time()
for random_number in [100]:#,2024,800,500,705,2023,2005,208,803,42,36]:
    
    regr = RandomForestRegressor(n_estimators=50,max_depth=4, random_state = random_number)
    k = 5
    n_fold = int(np.floor(data_X.shape[0]/k))
    indx0 = {}
    indx1 = {}
    indx2 = {}
    mse = np.zeros((k,1))
    Main_mse = np.zeros((k,1))
    Main_mean_per = np.zeros((k,1))
    Main_var_per = np.zeros((k,1))
    Main_mean = np.zeros((k,1))
    Main_var = np.zeros((k,1))
    PCC = np.zeros((k,1))
    AUC = np.zeros((k,1))
    ACC = np.zeros((k,1))
    indx3 = np.zeros((k,1))
    indx4 = np.zeros((k,1))
    f1_score_0 = np.zeros((k,1))
    f1_score_1 = np.zeros((k,1))
    for i in range(0,k):
        
        val_range = range(n_fold*i,n_fold*(i+1)) 
        
        
        X_val = data_X[val_range,:]
        y_val = np.ravel(y_antibody[val_range])
        
    
        X_train =  np.delete(data_X, val_range, axis = 0)
        y_train =  np.ravel(np.delete(y_antibody, val_range, axis = 0))
        regr.fit(X_train, y_train)
        y_val_perdicted = regr.predict(X_val)
        y_train_per = regr.predict(X_train)
        Main_mse[i,0] = np.mean(np.square(y_val_perdicted-y_val))
        Main_mean_per[i,0] = np.mean(y_val_perdicted)
        Main_var_per[i,0] = np.var(y_val_perdicted)
        Main_mean[i,0] = np.mean(y_val)
        Main_var[i,0] = np.var(y_val)
        y_train_per = (y_train_per - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376)
    
        y_val_perdicted = (y_val_perdicted - 0.1)/0.8*(6.056981036668113-(-0.3979400086720376))+(-0.3979400086720376)
    
        y_val = (y_val - 0.1)/0.8*5.87732526443497
        y_train = (y_train - 0.1)/0.8*5.87732526443497
    
        # print('Mean absolute error_' + str(i) )
        # print(mean_absolute_error(y_val, y_val_perdicted))
        # print('Mean square error_' + str(i) )
        # print(mean_squared_error(y_val, y_val_perdicted))
        # print('***************************************************')
    
        popt, pcov = curve_fit(f,  y_train,  y_train - y_train_per ) # your data x, y to fit
    
        indx00,indx11,indx22,indx3[i,0],indx4[i,0],mse[i,0],PCC[i,0] = Blant_Alman_plot(y_val,y_val_perdicted + f(y_val_perdicted, popt[0],popt[1]),'log(Antibodty)',i)
        
        y_val_perdicted = y_val_perdicted +f(y_val_perdicted, popt[0],popt[1])
        y_val[y_val < np.log10(80)] = 0
        y_val[y_val >= np.log10(80)] = 1
        y_val_perdicted[y_val_perdicted < np.log10(80)] = 0
        y_val_perdicted[y_val_perdicted >= np.log10(80)] = 1
        AUC[i,0] = roc_auc_score(y_val, y_val_perdicted)
        ACC[i,0] =accuracy_score(y_val, y_val_perdicted)
        f1_score_1[i,0] = f1_score(y_val, y_val_perdicted, pos_label=1)
        f1_score_0[i,0] = f1_score(y_val, y_val_perdicted, pos_label=0)
        # print('***************************************************')
        # print('Classification Report_' + str(i))
        # print('***************************************************')
        # print(classification_report(np.ravel(y_val), np.ravel(y_val_perdicted), labels = [0,1], target_names=['not Immune','Immune']))
        # print('***************************************************')
    
    elapsed_time = time.time()-start_time
    tmp0 = np.mean(mse)
    tmp1 = np.std(mse)
        
    tmp3 = np.mean(PCC)
    tmp4= np.std(PCC)
    
    tmp5 = np.mean(indx3)
    tmp6 = np.std(indx3)
    
    tmp15 = np.mean(indx4)
    tmp16 = np.std(indx4)
    
    
    tmp7 = np.mean(AUC)
    tmp8 = np.std(AUC)
    
    tmp9 = np.mean(ACC)
    tmp10 = np.std(ACC)
    
    tmp11 = np.mean(f1_score_0)
    tmp12 = np.std(f1_score_0)
    
    tmp13 = np.mean(f1_score_1)
    tmp14 = np.std(f1_score_1)
    
    
    from tabulate import tabulate
    print(tabulate([[random_number,
                     f'{np.mean(Main_mse):.2f} +- {np.std(Main_mse):.2f}',
                     f'{np.mean(Main_mean_per):.2f} +- {np.std(Main_mean_per):.2f}',
                     f'{np.mean(Main_var_per):.2f} +- {np.std(Main_var_per):.2f}',
                     f'{np.mean(Main_mean):.2f} +- {np.std(Main_mean):.2f}',
                     f'{np.mean(Main_var):.2f} +- {np.std(Main_var):.2f}',
                     f'{tmp0:.2f} +- {tmp1:.2f}',
                     f'{tmp3:.2f} +- {tmp4:.2f}',
                     f'{tmp5:.2f} +- {tmp6:.2f}',
                     f'{tmp15:.2f} +- {tmp16:.2f}',
                     f'{tmp7:.2f} +- {tmp8:.2f}',
                     f'{tmp9:.2f} +- {tmp10:.2f}',
                     f'{tmp11:.2f} +- {tmp12:.2f}',
                     f'{tmp13:.2f} +- {tmp14:.2f}']], 
                    headers = ['Random Number',
                               'Main MSE',
                               'Mean_per',
                               'var_per',
                               'Mean',
                               'var',
                               'MSE',
                               'Pearson_cor_coef',
                               'dist_more_65',
                               'dist_more_37.5',
                               'AUC',
                               'ACC',
                               'f1 not immune',
                               'f1 immune'],tablefmt='orgtbl'))








df = pd.DataFrame(data_X, columns = feature_names_longformat)

explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(data_X)
plt.figure()
shap.summary_plot(shap_values, df.values, plot_type="bar", class_names= ['immune'], feature_names = df.columns)
plt.figure()
shap.summary_plot(shap_values, df.values, feature_names = df.columns)



mean_Shap_values = np.mean(np.abs(shap_values), axis = 0)
importances = pd.Series(mean_Shap_values, index=feature_names_longformat)
importances = pd.DataFrame(importances, columns = ['rank'])
importances = importances.sort_values("rank", ascending=False)
fig, ax = plt.subplots()

color = (0.7843137254901961, # red
0.8784313725490196, # green
0.7058823529411765, # blue
1) # transparency  
# Horizontal Bar Plot
ax.barh(importances.index[0:15],importances['rank'][0:15].values,color=color,edgecolor =[0,0,0],linewidth=1)
# Remove axes splines
for s in ['top','right']: 
    ax.spines[s].set_visible(False)
# Remove y Ticks
ax.yaxis.set_ticks_position('none')
# Show top values
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Feature importance',loc ='left', )
plt.xlabel('Shap Values')
# Show Plot
plt.show()    
    
    

#filename = 'D:\\UHN\\Covid19 Vaccination\\Github_submission\\Random_forest_weights\\random_forest'
#with open(filename, 'wb') as f:
 #   cPickle.dump(regr, f)


  



path2data = 'C:\\Ghazal\\Covid19 Vaccination\\'
#fname = 'SOT_COVID_Data_Long_6M_padded_event_days.npz'
fname = 'SOT_COVID_Data_Long_20240102.npz'
# fname = 'SOT_COVID_Data_Long_20231121.npz'

data = np.load(path2data + fname)    



X = data['X'][:,1:40]
y = data['y']
Antibody = data['y_value']

X = np.reshape(X,[303,5,39])
X = np.delete(X, 23, axis = 2)

Antibody = np.reshape(Antibody,[303,5])


y_antibody = np.zeros((len(y),1))
X_12M = np.zeros((303,38))
n = 0
j = 0
for i in range(0,len(Antibody)):
    if np.isinf(Antibody[i,4]):
        j = j + 1
    else:
        y_antibody[n] = Antibody[i,4]
        X_12M[n,:] = X[i,2,:]
        X_12M[n,19] = X[i,0,19]
        n = n + 1

y_antibody = y_antibody[0:n]
X_12M = X_12M[0:n,:]





selection_index = np.array(range(0,303))
        


y_antibody = np.ravel(y_antibody)

data_X = X_12M
#(y_antibody > 80) & (y_antibody < 5000)
# data_X = X_12M[(y_antibody > 80) & (y_antibody < 5000),:]
#P_ID = Patient_ID[selection_index[(y_antibody > 80) & (y_antibody < 5000)].astype(int)]
# import random
# indx = random.randint(0, data_X.shape[0])

df = pd.DataFrame(data_X, columns = feature_names_longformat)

for i in [26]:#[18,82,174]:
    #print(PX_ID[i])
    print('*****************************')
    choosen_instance = df.loc[[i]]
    shap_values = explainer.shap_values(choosen_instance)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, choosen_instance,  matplotlib = True, show = True,contribution_threshold=0.05)
    # xx, locs = plt.xticks()
    # ll = ['%.3f' % a for a in xx]
    # plt.xticks(xx, ll)
    # plt.show()
# explainer = shap.Explainer(rf)
# shap_values = explainer(pos_dataset)
# shap.plots.waterfall(shap_values[1][:,1], max_display=20)

