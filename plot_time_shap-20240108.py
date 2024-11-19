# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:49:15 2023

@author: azarf
"""
import pandas as pd
from matplotlib import pyplot as plt
from My_beeswarm import My_beeswarm
import numpy as np
from feature_names_longformat import feature_names_longformat
from shap.plots import colors
from shap.plots._labels import labels
import tensorflow as tf
from LSTM_wtih_DynamicRouting import LSTM_DyanRout_model
from feature_names_longformat import plot_feats



#path2LSTM_with_Dyna_Routing = 'D:\\UHN\\Covid19 Vaccination\\Github_submission\\LSTM_DynamicRouting_weights_20240322\\'
#model = LSTM_DyanRout_model
#model.load_weights(path2LSTM_with_Dyna_Routing)    


path2LSTM = 'C:\\Users\\azarf\\Downloads\\LSTM\\'
LSTM = tf.keras.models.load_model(path2LSTM)


#model entry point
f = lambda x: model.predict(x)
#read features
path2data = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_Long_20240102.npz'

data = np.load(path2data + fname)    




X = data['X'][:,1:40]
X = np.delete(X, 14, axis = 1)
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
y_12M_class[y_12M_class == 1] = 0
y_12M_class[y_12M_class == 2] = 1

y_12M = np.ravel(y_12M[0:n,0])
X_12M = X_12M[0:n,:,:]


model_features = feature_names_longformat
time_point = 5
feature_size = X_12M.shape[2]
sample_count = n



df = pd.DataFrame(np.reshape(X_12M,(303*5,38)), columns = feature_names_longformat)
df['label'] = np.repeat(y_12M_class,(5))
df['activity'] = None
df.loc[df['label'] == 1, ['activity']] = 'immune'
df.loc[df['label'] == 0, ['activity']] = 'not immune'

df['id'] = np.repeat(np.array(range(0,sample_count)),(5))
df['timestamp'] = np.tile(np.transpose(np.array(range(0,5))),(sample_count))
df['all_id'] =np.array(range(0,sample_count*5))
time_feat = 'timestamp'
label_feat = 'label'
sequence_id_feat = 'id'



from timeshap.explainer import local_report

path2data = 'D://UHN//Covid19 Vaccination//Github_submission//LSTM_TimeSHAP_20240322//'
fname = 'feature_all_tf.csv'
df_feature = pd.read_csv(path2data + fname)
pos_dataset = df[df['label'] == 1]
ids_for_test = df_feature['Entity'].unique()
plot_dataset = pos_dataset.loc[pos_dataset['id'] == ids_for_test[0]]


for i in range(1,len(ids_for_test)):
    plot_dataset = plot_dataset.append(pos_dataset.loc[pos_dataset['id'] == ids_for_test[i]])
    
plot_dataset = plot_dataset.drop(columns=['label','id', 'activity','timestamp','all_id'])
features_raw = plot_dataset.values
features_raw = np.reshape(features_raw,[int(len(plot_dataset)/time_point),time_point,feature_size])
cell_shap = np.zeros((len(plot_dataset),feature_size))

for i in range(0,int(len(plot_dataset)/5)):
    print('from ' + str(i*5) + ' to ' + str((i+1)*5))
    # select model features only
    pos_x_data = plot_dataset.iloc[i*5:(i+1)*5]
    pos_x_data = pos_x_data[model_features]
    # convert the instance to numpy so TimeSHAP receives it
    pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)



    #Local Report on positive instance
    
    rng = np.random.default_rng()
    background = rng.random((5, feature_size))


    pruning_dict = {'tol': 0.000}
    event_dict = {'rs': 42, 'nsamples': 300}
    feature_dict = {'rs': 42, 'nsamples': 300, 'feature_names': model_features, 'plot_features': plot_feats}
    cell_dict = {'rs': 42, 'nsamples':300, 'top_x_feats': feature_size, 'top_x_events': feature_size}

    cell_level = local_report(f, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid=i, entity_col='id', baseline=background)
    cell_level = cell_level.sort_index(axis=0)
    tmp = cell_level['Shapley Value'].values
    tmp = np.reshape(tmp,(5,feature_size))
    cell_shap[i*5:(i+1)*5,:] = tmp
    
myfeatures = np.zeros((int(len(plot_dataset)/time_point),feature_size))
    
for i in range(0,239):
    for j in range(0,feature_size):
        index = np.argmax(np.abs(cell_shap[i*5:(i+1)*5,j]))
        myfeatures[i,j] = features_raw[i,index,j]
#Read features


df_feature = df_feature.drop(columns=['Entity', 'Tolerance'])


Shap_values = np.reshape(df_feature['Shapley Value'].values,(239,feature_size))
# Shap_values = np.repeat(Shap_values, repeats=5, axis=0)

Shap_values_rand = np.zeros(cell_shap.shape)

for i in range(0,len(cell_shap)): 
    Shap_values_rand[i,:] = cell_shap[i,:] + np.random.rand(1,feature_size)*cell_shap[i,:]*0.1

plt.figure()
mean_Shap_values, order = My_beeswarm(Shap_values,cell_shap, max_display=20,
             features=plot_dataset.values, feature_names =feature_names_longformat, color=colors.red_blue,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, color_bar_label=labels["FEATURE_VALUE"])



importances = pd.Series(mean_Shap_values, index=feature_names_longformat)
importances = pd.DataFrame(importances, columns = ['rank'])
importances = importances.sort_values("rank", ascending=False)




fig, ax = plt.subplots(figsize =(16, 9))

# Horizontal Bar Plot
ax.barh(np.flip(importances.index[0:20]),np.flip(importances['rank'][0:20].values))
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
 
# Add Plot Title
ax.set_title('Sports car and their price in crore',
             loc ='left', )
 
# Add Text watermark
fig.text(0.9, 0.15, 'Jeeteshgavande30', fontsize = 12,
         color ='grey', ha ='right', va ='bottom',
         alpha = 0.7)
 
# Show Plot
plt.show()


LSTM_feature_importance = {}
for i in range(0,20):
    LSTM_feature_importance[importances.index[i]] = importances['rank'][0:20].values[i]
    