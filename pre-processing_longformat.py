# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:54:05 2023

@author: Ghazal Azarfar
"""
# %matplotlib qt 


##############################################################################
#Initialization
##############################################################################
import pandas as pd
import numpy as np
from utility import normalize as normalize
from utility import normalize_categorized as normalize_categorized
from utility import days_between_dates as days_between_dates
from utility import data_stat as data_stat
import matplotlib.pyplot as plt
from feature_names_longformat import feature_names_longformat


##############################################################################
# Read Data
##############################################################################
path2data = "C:\\Ghazal\\Covid19 Vaccination\\"
fname = 'REDCap Extraction OCT 17 2023 with RBD data.xlsx'
df0 = pd.read_excel(path2data + fname, header=[0, 1])

Header = 'Antibody Information'
columns = [#'6 month follow-up Antibody results (U/ml)',#'Post-fourth dose Antibody results (U/ml)',
          '12 month follow-up Antibody results (U/ml)']


df0['Include'] = np.zeros((len(df0),1))

for i in range(0, len(columns)):
    df0.loc[df0[Header,columns[i]] == '<0.400', (Header,columns[i])] = 0.4
    df0.loc[df0[Header,columns[i]] == '<0.4', (Header,columns[i])] = 0.4
    df0.loc[df0[Header,columns[i]].isna(), [[Header,columns[i]]]]= 0
    df0['Include'] = df0['Include'].values + df0[Header,columns[i]].values


df = df0.loc[df0['Include'] !=0]
# Exclude Patients
##############################################################################
Exclude_patients = [342,343,344]
df.drop(labels = Exclude_patients, axis = 0, inplace = True)

print(len(df))


V = ['Visit 1 First Dose','Visit 2 Second Dose','Third Dose Information','6M Visit','12M Visit']
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
Dose = ['Prednisone Dose (mg)','Tacrolimus Dose (mg)','Mycophenolate mofetil or mycophenolate sodium Dose (mg)']
Frequency = ['Prednisone Frequency','Tacrolimus Frequency','Mycophenolate mofetil or mycophenolate sodium Frequency']
Immuno = [Medication,Dose,Frequency]    

for k in [0,4]:#range(0,len(V)-1):
    for j in range(0,len(Immuno)):
        for i in range(0,len(Immuno[j])):       
            # print(V[k] + ': (' + Immuno[j][i] + ') :' + str(df[V[k],Immuno[j][i]].isna().sum()))
            # print(V[k+1] + ': (' + Immuno[j][i] + ') :' + str(df[V[k+1],Immuno[j][i]].isna().sum()))
            # condition = ((df[V[k+1],'Any changes in immunosuppression medication since last visit?'] == 'No') | (df[V[k+1],'Any changes in immunosuppression medication since last visit?'].isna()) & (df[V[k+1],Immuno[j][i]].isna()) & (df[V[k],Immuno[j][i]].notna()))
            # df.loc[condition ,[[V[k+1],Immuno[j][i]]]] = df.loc[condition ,[[V[k],Immuno[j][i]]]].values
            # print(V[k] + ': (' + Immuno[j][i] + ') :' + str(df[V[k],Immuno[j][i]].isna().sum()))
            # print(V[k+1] + ': (' + Immuno[j][i] + ') :' + str(df[V[k+1],Immuno[j][i]].isna().sum()) + '(' + str(df[V[k+1],Immuno[j][i]].isna().sum()/303*100) + '%)')
            print('*************************************************************')
            print('CHUM-021')
            print('*************************************************************')
            print(V[k] + ': (' + Immuno[j][i] + ') :' + str(df.loc[df['Characteristics','Patient ID'] == 'CHUM-021', [[V[k],Immuno[j][i]]]]))

##############################################################################
# Labels
##############################################################################
columns = ['Post-first dose Antibody results (U/ml)',
           'Post-second dose Antibody results (U/ml)',
           'Post-third dose Antibody results (U/ml)',
           '6 month follow-up Antibody results (U/ml)',
           #'Post-fourth dose Antibody results (U/ml)',
          '12 month follow-up Antibody results (U/ml)']

labels = np.zeros((len(df),len(columns)))
Antibody = np.zeros((len(df),len(columns)))


for i in range(0, len(columns)):
    df.loc[df[Header,columns[i]] == '<0.400', (Header,columns[i])] = 0.4
    df.loc[df[Header,columns[i]] == '<0.4', (Header,columns[i])] = 0.4
    df.loc[df[Header,columns[i]].isna(), [[Header,columns[i]]]]= 0
    Antibody[:,i] = df[Header, columns[i]].values
    
Antibody = Antibody[~np.all(Antibody == 0, axis=1)]
labels = Antibody.copy()
labels[labels == 0] = 100
labels[labels < 80] = 1
labels[labels == 100] = 0
labels[labels >= 80] = 2

Header = 'Antibody Information'
# columns = ['Post-first dose Antibody results (U/ml)','Post-second dose Antibody results (U/ml)',
#           'Post-third dose Antibody results (U/ml)','6 month follow-up Antibody results (U/ml)',#'Post-fourth dose Antibody results (U/ml)',
#           '12 month follow-up Antibody results (U/ml)']

# count = np.zeros((3,labels.shape[1]))
# for i in range(0,labels.shape[1]):
#     tmp = labels[:,i]
#     for j in range(0,3):
#         count[j,i] = len(tmp[tmp == j])

# time_point = ('First vaccination', 'Second vaccination', 'third vaccination', '6 months follow-up','fourth vaccination','1 year follow-up')
# state_counts = {
#     'immune': count[2,:],
#     'Nonimmune': count[1,:],
#     'Missing': count[0,:],
# }
# width = 0.6  # the width of the bars: can also be len(x) sequence


# fig, ax = plt.subplots()
# bottom = np.zeros(labels.shape[1])

# for state, state_count in state_counts.items():
#     p = ax.bar(time_point, state_count, width, label=state, bottom=bottom)
#     bottom += state_count

#     ax.bar_label(p, label_type='center')

# ax.set_title('Antibody availability')
# ax.legend()

# plt.show()

    
#figure
# import matplotlib.plt as plt

# plt.figure()
# for i in range(0,6): 
#     plt.subplot(3,2,i+1)
#     plt.scatter(range(0,len(Antibody)),np.log10(Antibody[:,i]))
#     plt.title('visit ' + str(i))
#     plt.ylabel('Log(Antibody)')
    
    
permanent_feature_length = 19
visit_feature_size = 21
permanent_data = np.zeros((len(df),permanent_feature_length)) # define an input variable for the predictive model
visit_1_data = np.zeros((len(df),visit_feature_size)) # define an input variable for the predictive model
visit_2_data = np.zeros((len(df),visit_feature_size)) # define an input variable for the predictive model
visit_3_data = np.zeros((len(df),visit_feature_size)) # define an input variable for the predictive model
visit_3_data = np.zeros((len(df),visit_feature_size)) # define an input variable for the predictive model
Six_months_data = np.zeros((len(df),visit_feature_size))
Towelve_months_data = np.zeros((len(df),visit_feature_size))
##############################################################################
# Patients Characteristics
##############################################################################
permanent_data[:,0] = normalize_categorized(df['Characteristics','Data Access Group'])
permanent_data[:,1]  =df['Characteristics','Age'] #normalize(df['Characteristics','Age'])
permanent_data[:,2]  = normalize(df['Characteristics','BMI'])
permanent_data[:,3]= normalize_categorized(df['Characteristics','Sex'])


##############################################################################
# Organ Transplanted
##############################################################################
Organ_Transplanted = ['Lung','Heart','Liver','Kidney','Pancreas']

indx = 4
for i in range(0,len(Organ_Transplanted)):
    column = 'Organ Transplanted (check all that apply) (choice=' + Organ_Transplanted[i] + ')'
    df.loc[df['Characteristics',column].isna(), [['Characteristics',column]]]= 'MyNaN'
    permanent_data[:,indx] = normalize_categorized(df['Characteristics',column])
    indx = indx + 1


##############################################################################
#Transplant Info
##############################################################################
column = ['Re-transplant?','Transplant Induction ','Drug of induction','Treatment for rejection (in the past 3 months)',
          'Does the patient have chronic kidney disease (defined as eGFR < 30)',
          'Does the patient have any other immunosuppressive conditions (i.e. HIV, concurrent chemotherapy, etc)'
          #'Type of HSCT transplant  (choice=Allogeneic- matched sibling)',
          # 'Type of HSCT transplant  (choice=Allogeneic- matched unrelated)',
          # 'Type of HSCT transplant  (choice=Allogeneic- haploidentical)',
          #  'Type of HSCT transplant  (choice=Allogeneic- mismatched (non-haplo))',
          #  'Type of HSCT transplant  (choice=Autologous)'
           ]



for i in range(0,len(column)):
    df.loc[df['Characteristics',column[i]].isna(), [['Characteristics',column[i]]]]= 'MyNaN'
    permanent_data[:,indx] = normalize_categorized(df['Characteristics',column[i]])
    indx = indx+1


# ##############################################################################
# #GVHD Info
# ##############################################################################
# GVHD = ['Anti-thymocyte globulin','Post-transplant cyclophosphamide','Cyclosporine','Tacrolimus','MMF','Sirolimus','Other']



# for i in range(0,len(GVHD)):  
#     column = 'GVHD Prophylaxis (choice=' + GVHD[i] + ')'
#     df.loc[df['Characteristics',column].isna(),[['Characteristics',column]]]= 'MyNaN'
#     permanent_data[:,indx] = normalize_categorized(df['Characteristics',column])
#     indx = indx + 1

##############################################################################
# COVID Information
##############################################################################
df.loc[df['COVID Information', 'Did patient contract COVID-19 at any time during the study?'].isna(), [['COVID Information', 'Did patient contract COVID-19 at any time during the study?']]]= 'MyNaN'
permanent_data[:,indx] = normalize_categorized(df['COVID Information', 'Did patient contract COVID-19 at any time during the study?'])
indx = indx + 1



df.loc[df['COVID Information', 'post COVID Antibody results U/ml'].isna(), ('COVID Information', 'post COVID Antibody results U/ml')]= 0
permanent_data[:,indx] = df['COVID Information', 'post COVID Antibody results U/ml']
permanent_data[:,indx]  = permanent_data[:,indx] /np.max(permanent_data[:,indx])
indx = indx + 1




df.loc[df['Antibody Information', 'Pre-vaccine Antibody results (U/ml)'] == '<0.400', ('Antibody Information', 'Pre-vaccine Antibody results (U/ml)')] = 0.4
df.loc[df['Antibody Information', 'Pre-vaccine Antibody results (U/ml)'] == '<0.4', ('Antibody Information', 'Pre-vaccine Antibody results (U/ml)')] = 0.4
df.loc[df['Antibody Information', 'Pre-vaccine Antibody results (U/ml)'].isna(), ('Antibody Information', 'Pre-vaccine Antibody results (U/ml)')]= 0
permanent_data[:,indx] = df['Antibody Information', 'Pre-vaccine Antibody results (U/ml)']
permanent_data[:,indx]  = permanent_data[:,indx] /np.max(permanent_data[:,indx])
indx = indx + 1

df.loc[df['Fourth Dose', 'Did the patient receive a fourth dose?'].isna(), [['Fourth Dose', 'Did the patient receive a fourth dose?']]]= 'MyNaN'
permanent_data[:,indx] = normalize_categorized(df['Fourth Dose', 'Did the patient receive a fourth dose?'])
indx = indx + 1

print('permanent data index is:' + str(indx))


##############################################################################
# Visit first Info
##############################################################################
#Dose 1
indx = 0
visit_1_data[:,indx] = np.zeros((len(df),))+1
indx = indx + 1

Events = ['Transplant date','Date of COVID-19 Vaccine Dose 1 date' ,'Vaccine dose 2 date', 'Date of third dose\xa0', 'Date of fourth dose\xa0']
Header_row = ['Characteristics','Characteristics','Characteristics', 'Third Dose Information','Fourth Dose' ]
Vaccination_Event_Time_Interval = ['Tran to first dose days','first to second dose days', 'second_to_third_dose_days' , 'third_to_fourth_dose_days']
 


i = 0
day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval[i]]
df[day_column_name[0],day_column_name[1]] = None
header_date1 = [Header_row[i],Events[i]]
header_date2 = [Header_row[i+1],Events[i+1]]
days, count = days_between_dates(df, header_date1, header_date2, day_column_name)
tmp = pd.DataFrame(days, columns = [Vaccination_Event_Time_Interval[i]])
day_min, day_max, day_mean, day_var, day_filled = data_stat(tmp[Vaccination_Event_Time_Interval[i]], 1,Vaccination_Event_Time_Interval[i])
days[np.isnan(days)] = 0
days[days < 0] = 0
minval = np.min(days[np.nonzero(days)])
maxval = np.max(days)
days = (days - minval)*0.8/(maxval-minval)+0.1
visit_1_data[:,indx]  = days.flatten()
indx = indx + 1



vaccine_dose =  'Vaccine administered first dose'
vaccine_header = 'Characteristics'
df.loc[df[vaccine_header, vaccine_dose].isna(), [[vaccine_header, vaccine_dose]]]= 'MyNaN'
tmp = df[vaccine_header, vaccine_dose]
tmp = tmp.replace('MyNaN', 0) 
tmp = tmp.replace('UNK', 0)
tmp = tmp.replace('Moderna', 1) 
tmp = tmp.replace('Pfizer', 2) 
tmp = tmp.replace('Astra Zeneca', 3) 
visit_1_data[:,indx] = tmp.values
indx = indx + 1

#Did the patient receive a different vaccine for the first dose?
visit_1_data[:,indx] = np.zeros((len(df),))
indx = indx + 1


df.loc[df['Visit 1 First Dose','Hospitalizations'].isna(), [['Visit 1 First Dose','Hospitalizations']]]= 'MyNaN'
visit_1_data[:,indx]  = normalize_categorized(df['Visit 1 First Dose','Hospitalizations'])
indx = indx + 1

df.loc[df['Visit 1 First Dose','Documented COVID-19 infection'].isna(), [['Visit 1 First Dose','Documented COVID-19 infection']]]= 'MyNaN'
visit_1_data[:,indx] = normalize_categorized(df['Visit 1 First Dose','Documented COVID-19 infection'])
indx = indx + 1

df.loc[df['Visit 1 First Dose','Rejection since last visit\xa0'].isna(), [['Visit 1 First Dose','Rejection since last visit\xa0']]]= 'MyNaN'
visit_1_data[:,indx]  = normalize_categorized(df['Visit 1 First Dose','Rejection since last visit\xa0'])
indx = indx + 1

#Changes in immunosupression medication
visit_1_data[:,indx] = np.zeros((len(df),))
indx = indx + 1


headr = 'Visit 1 First Dose'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    visit_1_data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1

# headr = 'Visit 1 First Dose'
# column = 'More than one medication?'
# df[headr,column] = None
# Med = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
# tmp0 = np.zeros((len(df),len(Med)))
# for i in range(0,len(Med)): 
#     df.loc[df[headr,Med[i]].isna(), [[headr,Med[i]]]]= 'MyNaN'
#     tmp = df[headr, Med[i]]
#     tmp = tmp.replace('MyNaN', 0) 
#     tmp = tmp.replace('No', 1) 
#     tmp = tmp.replace('Yes', 2)     
# #More than one medication
# tmp1 = np.sum(tmp0, axis = 1)
# tmp1[tmp1 < 2] = 1

#     tmp0[:,i] = tmp.values
# tmp1[tmp1 >= 2] = 2
# visit_1_data[:,indx]  = tmp1
# indx = indx + 1



headr = 'Visit 1 First Dose'
Dose = ['Prednisone Dose (mg)',
        'Tacrolimus Dose (mg)',
        'Most recent Tac level',
        'Mycophenolate mofetil or mycophenolate sodium Dose (mg)']



for i in range(0,len(Dose)):
    df.loc[df[headr,Dose[i]].isna(), [[headr,Dose[i]]]]= 0
    visit_1_data[:,indx]  = normalize(df[headr,Dose[i]].astype(np.float32))
    indx = indx + 1



headr = 'Visit 1 First Dose'
Frequency = ['Prednisone Frequency',
             'Tacrolimus Frequency',
             'Mycophenolate mofetil or mycophenolate sodium Frequency']
    
for i in range(0,len(Frequency)):
     df.loc[df[headr,Frequency[i]].isna(), [[headr,Frequency[i]]]]= 'MyNaN'
     tmp = df[headr, Frequency[i]]
     tmp = tmp.replace('MyNaN', 0) 
     tmp = tmp.replace('UNK', 0)
     tmp = tmp.replace('EOD', 1) #every othr day
     tmp = tmp.replace('Daily', 2) # once a day
     tmp = tmp.replace('BID', 3) #two times a day
     tmp = tmp.replace('TID',4) #three times a day
     visit_1_data[:,indx]  =  tmp.values
     indx = indx + 1

print('visit 1 index is:' + str(indx))

##############################################################################
# Visit Second Dose
##############################################################################
#Dose 2
indx = 0
visit_2_data[:,indx] = np.zeros((len(df),)) + 2
indx = indx + 1





Events = ['Date of COVID-19 Vaccine Dose 1 date','Vaccine dose 2 date']
Header_row = ['Characteristics','Characteristics']
Vaccination_Event_Time_Interval = 'first to second dose days'


day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval]
df[day_column_name[0],day_column_name[1]] = None
header_date1 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date' ]
header_date2 = ['Characteristics','Vaccine dose 2 date']
days, count = days_between_dates(df, header_date1, header_date2, day_column_name)
tmp = pd.DataFrame(days, columns = [Vaccination_Event_Time_Interval])
day_min, day_max, day_mean, day_var, day_filled = data_stat(tmp[Vaccination_Event_Time_Interval], 1,Vaccination_Event_Time_Interval)
days[np.isnan(days)] = 0
days[days < 0] = 0
minval = np.min(days[np.nonzero(days)])
maxval = np.max(days)
days = (days - minval)*0.8/(maxval-minval)+0.1
visit_2_data[:,indx]  = days.flatten()
indx = indx + 1



vaccine_dose =  'Vaccine administered second dose'
vaccine_header = 'Characteristics'
df.loc[df[vaccine_header, vaccine_dose].isna(), [[vaccine_header, vaccine_dose]]]= 'MyNaN'
tmp = df[vaccine_header, vaccine_dose]
tmp = tmp.replace('MyNaN', 0) 
tmp = tmp.replace('UNK', 0)
tmp = tmp.replace('Moderna', 1) 
tmp = tmp.replace('Pfizer', 2) 
tmp = tmp.replace('Astra Zeneca', 3) 
visit_2_data[:,indx] = tmp.values
indx = indx + 1

df.loc[df['Characteristics', 'Did the patient receive a different vaccine for the second dose?'].isna(), [['Characteristics', 'Did the patient receive a different vaccine for the second dose?']]]= 'No'
visit_2_data[:,indx] = normalize_categorized(df['Characteristics', 'Did the patient receive a different vaccine for the second dose?'])
indx = indx + 1


df.loc[df['Visit 2 Second Dose', 'Hospitalizations'].isna(), [['Visit 2 Second Dose', 'Hospitalizations']]]= 'No'
visit_2_data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Hospitalizations'])
indx = indx + 1

df.loc[df['Visit 2 Second Dose', 'Documented COVID-19 infection'].isna(), [['Visit 2 Second Dose', 'Documented COVID-19 infection']]]= 'No'
visit_2_data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Documented COVID-19 infection'])
indx = indx + 1

df.loc[df['Visit 2 Second Dose', 'Rejection since last visit\xa0'].isna(), [['Visit 2 Second Dose', 'Rejection since last visit\xa0']]]= 'No'
visit_2_data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Rejection since last visit\xa0'])
indx = indx + 1


df.loc[df['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?'].isna(), [['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?']]]= 'No'
visit_2_data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1



headr = 'Visit 2 Second Dose'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    visit_2_data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1


# headr = 'Visit 2 Second Dose'
# column = 'More than one medication?'
# df[headr,column] = None
# Med = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
# tmp0 = np.zeros((len(df),len(Med)))
# for i in range(0,len(Med)): 
#     df.loc[df[headr,Med[i]].isna(), [[headr,Med[i]]]]= 'MyNaN'
#     tmp = df[headr, Med[i]]
#     tmp = tmp.replace('MyNaN', 0) 
#     tmp = tmp.replace('No', 1) 
#     tmp = tmp.replace('Yes', 2) 
#     tmp0[:,i] = tmp.values

# #more than one medication
# tmp1 = np.sum(tmp0, axis = 1)
# tmp1[tmp1 < 2] = 1
# tmp1[tmp1 >= 2] = 2
# visit_2_data[:,indx]  = tmp1
# indx = indx + 1


headr = 'Visit 2 Second Dose'
Dose = ['Prednisone Dose (mg)',
        'Tacrolimus Dose (mg)',
        'Most recent Tac level',
        'Mycophenolate mofetil or mycophenolate sodium Dose (mg)']



for i in range(0,len(Dose)):
    df.loc[df[headr,Dose[i]].isna(), [[headr,Dose[i]]]]= 0
    visit_2_data[:,indx]  = normalize(df[headr,Dose[i]].astype(np.float32))
    indx = indx + 1
    
    

headr = 'Visit 2 Second Dose'    
Frequency = ['Prednisone Frequency',
             'Tacrolimus Frequency',
             'Mycophenolate mofetil or mycophenolate sodium Frequency']    
    
for i in range(0,len(Frequency)):
     df.loc[df[headr,Frequency[i]].isna(), [[headr,Frequency[i]]]]= 'MyNaN'
     tmp = df[headr, Frequency[i]]
     tmp = tmp.replace('MyNaN', 0) 
     tmp = tmp.replace('UNK', 0)
     tmp = tmp.replace('EOD', 1) #every othr day
     tmp = tmp.replace('Daily', 2) # once a day
     tmp = tmp.replace('BID', 3) #two times a day
     tmp = tmp.replace('TID',4) #three times a day
     visit_2_data[:,indx]  = tmp.values
     indx = indx + 1

print('visit 2 index is:' + str(indx))

##############################################################################
# Third Dose Information
##############################################################################
# vaccine_dose =  ['Vaccine administered first dose', 'Vaccine administered second dose', 'Vaccine administered ', 'Vaccine administered ' ]
# vaccine_header = ['Characteristics','Characteristics', 'Third Dose Information','Fourth Dose' ]

indx = 0
visit_3_data[:,indx] = np.zeros((len(df),)) + 3
indx = indx + 1




Vaccination_Event_Time_Interval = 'Second to third dose days'


day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval]
df[day_column_name[0],day_column_name[1]] = None
header_date1 = ['Characteristics','Vaccine dose 2 date']
header_date2 = ['Third Dose Information','Date of third dose\xa0']
days, count = days_between_dates(df, header_date1, header_date2, day_column_name)
tmp = pd.DataFrame(days, columns = [Vaccination_Event_Time_Interval])
day_min, day_max, day_mean, day_var, day_filled = data_stat(tmp[Vaccination_Event_Time_Interval], 1,Vaccination_Event_Time_Interval)
days[np.isnan(days)] = 0
days[days < 0] = 0
minval = np.min(days[np.nonzero(days)])
maxval = np.max(days)
days = (days - minval)*0.8/(maxval-minval)+0.1
visit_3_data[:,indx]  = days.flatten()
indx = indx + 1



vaccine_dose =  'Vaccine administered third dose'
vaccine_header = 'Third Dose Information'

df.loc[df[vaccine_header, vaccine_dose].isna(),[[vaccine_header, vaccine_dose]]]= 'MyNaN'
tmp = df[vaccine_header, vaccine_dose]
tmp = tmp.replace('MyNaN', 0) 
tmp = tmp.replace('UNK', 0)
tmp = tmp.replace('Moderna', 1) 
tmp = tmp.replace('Pfizer', 2) 
tmp = tmp.replace('Astra Zeneca', 3) 
visit_3_data[:,indx] = tmp.values
indx = indx + 1

df.loc[df['Third Dose Information', 'Did the patient receive a different vaccine for the third dose?'].isna(), [['Third Dose Information', 'Did the patient receive a different vaccine for the third dose?']]]= 'No'
visit_3_data[:,indx] = normalize_categorized(df['Third Dose Information', 'Did the patient receive a different vaccine for the third dose?'])
indx = indx + 1

#Hospitalization
visit_3_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

#Documented COVID-19 infection
visit_3_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

#Rejection since last visit
visit_3_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

df.loc[df['Third Dose Information', 'Any changes in immunosuppression medication since last visit?'].isna(), [['Third Dose Information', 'Any changes in immunosuppression medication since last visit?']]]= 'No'
visit_3_data[:,indx] = normalize_categorized(df['Third Dose Information', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1






headr = 'Third Dose Information'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    visit_3_data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1



# headr = 'Third Dose Information'
# column = 'More than one medication?'
# df[headr,column] = None
# Med = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
# tmp0 = np.zeros((len(df),len(Med)))
# for i in range(0,len(Med)): 
#     df.loc[df[headr,Med[i]].isna(), [[headr,Med[i]]]]= 'MyNaN'
#     tmp = df[headr, Med[i]]
#     tmp = tmp.replace('MyNaN', 0) 
#     tmp = tmp.replace('No', 1) 
#     tmp = tmp.replace('Yes', 2) 
#     tmp0[:,i] = tmp.values

# tmp1 = np.sum(tmp0, axis = 1)
# tmp1[tmp1 < 2] = 1
# tmp1[tmp1 >= 2] = 2
# visit_3_data[:,indx]  = tmp1
# indx = indx + 1


Dose = ['Prednisone Dose (mg)',
        'Tacrolimus Dose (mg)',
        'Most recent Tac level',
        'Mycophenolate mofetil or mycophenolate sodium Dose (mg)']



for i in range(0,len(Dose)):
    df.loc[df[headr,Dose[i]].isna(), [[headr,Dose[i]]]]= 0
    visit_3_data[:,indx]  = normalize(df[headr,Dose[i]].astype(np.float32))
    indx = indx + 1
    
headr = 'Third Dose Information'
Frequency = ['Prednisone Frequency',
             'Tacrolimus Frequency',
             'Mycophenolate mofetil or mycophenolate sodium Frequency']
    
for i in range(0,len(Frequency)):
     df.loc[df[headr,Frequency[i]].isna(), [[headr,Frequency[i]]]]= 'MyNaN'
     tmp = df[headr, Frequency[i]]
     tmp = tmp.replace('MyNaN', 0) 
     tmp = tmp.replace('UNK', 0)
     tmp = tmp.replace('EOD', 1) #every othr day
     tmp = tmp.replace('Daily', 2) # once a day
     tmp = tmp.replace('BID', 3) #two times a day
     tmp = tmp.replace('TID',4) #three times a day
     visit_3_data[:,indx]  = tmp.values
     indx = indx + 1


print('visit 3 index is:' + str(indx))
##############################################################################
# 6 Months Visit
##############################################################################

indx = 0
Six_months_data[:,indx] = np.zeros((len(df),)) + 4
indx = indx + 1


Events = ['Transplant date', '6 month follow-up sample collection date']
Vaccination_Event_Time_Interval = 'Third dose to 6M visit'


day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval]
df[day_column_name[0],day_column_name[1]] = None
header_date1 = ['Third Dose Information','Date of third dose\xa0']
header_date2 = ['6M Visit' , '6 month follow-up sample collection date']
days, count = days_between_dates(df, header_date1, header_date2, day_column_name)
tmp = pd.DataFrame(days, columns = [Vaccination_Event_Time_Interval])
day_min, day_max, day_mean, day_var, day_filled = data_stat(tmp[Vaccination_Event_Time_Interval], 1,Vaccination_Event_Time_Interval)
days[np.isnan(days)] = 0
days[days < 0] = 0
minval = np.min(days[np.nonzero(days)])
maxval = np.max(days)
days = (days - minval)*0.8/(maxval-minval)+0.1
Six_months_data[:,indx]  = days.flatten()
indx = indx + 1

#vaccine adminestrated
Six_months_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

#Did the patient recieve a different vaccine
Six_months_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

df.loc[df['6M Visit','Hospitalizations'].isna(), [['6M Visit','Hospitalizations']]]= 'MyNaN'
Six_months_data[:,indx]  = normalize_categorized(df['6M Visit','Hospitalizations'])
indx = indx + 1


df.loc[df['6M Visit','Documented COVID-19 infection'].isna(), [['6M Visit','Documented COVID-19 infection']]]= 'MyNaN'
Six_months_data[:,indx] = normalize_categorized(df['6M Visit','Documented COVID-19 infection'])
indx = indx + 1

df.loc[df['6M Visit','Rejection since last visit\xa0'].isna(), [['6M Visit','Rejection since last visit\xa0']]]= 'MyNaN'
Six_months_data[:,indx]  = normalize_categorized(df['6M Visit','Rejection since last visit\xa0'])
indx = indx + 1

df.loc[df['6M Visit', 'Any changes in immunosuppression medication since last visit?'].isna(), [['6M Visit', 'Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
Six_months_data[:,indx] = normalize_categorized(df['6M Visit', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1


headr = '6M Visit'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    Six_months_data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1
    
# headr = '6M Visit'
# column = 'More than one medication?'
# df[headr,column] = None
# Med = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
# tmp0 = np.zeros((len(df),len(Med)))
# for i in range(0,len(Med)): 
#     df.loc[df[headr,Med[i]].isna(), [[headr,Med[i]]]]= 'MyNaN'
#     tmp = df[headr, Med[i]]
#     tmp = tmp.replace('MyNaN', 0) 
#     tmp = tmp.replace('No', 1) 
#     tmp = tmp.replace('Yes', 2) 
#     tmp0[:,i] = tmp.values

# tmp1 = np.sum(tmp0, axis = 1)
# tmp1[tmp1 < 2] = 1
# tmp1[tmp1 >= 2] = 2
# Six_months_data[:,indx]  = tmp1
# indx = indx + 1

headr = '6M Visit'
Dose = ['Prednisone Dose (mg)',
        'Tacrolimus Dose (mg)',
        'Most recent Tac level',
        'Mycophenolate mofetil or mycophenolate sodium Dose (mg)']



for i in range(0,len(Dose)):
    df.loc[df[headr,Dose[i]].isna(), [[headr,Dose[i]]]]= 0
    Six_months_data[:,indx]  = normalize(df[headr,Dose[i]].astype(np.float32))
    indx = indx + 1
    
headr = '6M Visit'    
Frequency = ['Prednisone Frequency',
             'Tacrolimus Frequency',
             'Mycophenolate mofetil or mycophenolate sodium Frequency']    
    
for i in range(0,len(Frequency)):
     df.loc[df[headr,Frequency[i]].isna(), [[headr,Frequency[i]]]]= 'MyNaN'
     tmp = df[headr, Frequency[i]]
     tmp = tmp.replace('MyNaN', 0) 
     tmp = tmp.replace('UNK', 0)
     tmp = tmp.replace('EOD', 1) #every othr day
     tmp = tmp.replace('Daily', 2) # once a day
     tmp = tmp.replace('BID', 3) #two times a day
     tmp = tmp.replace('TID',4) #three times a day
     Six_months_data[:,indx]  = tmp.values
     indx = indx + 1

print('visit 6M index is:' + str(indx))
##############################################################################
# 12 Months Visit
##############################################################################
indx = 0
Towelve_months_data[:,indx] = np.zeros((len(df),)) + 5
indx = indx + 1



Vaccination_Event_Time_Interval = '6M to 12M visit'


day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval]
df[day_column_name[0],day_column_name[1]] = None
header_date1 = ['6M Visit' , '6 month follow-up sample collection date']
header_date2 = ['12M Visit','12 month follow-up sample collection date']
days, count = days_between_dates(df, header_date1, header_date2, day_column_name)
tmp = pd.DataFrame(days, columns = [Vaccination_Event_Time_Interval])
day_min, day_max, day_mean, day_var, day_filled = data_stat(tmp[Vaccination_Event_Time_Interval], 1,Vaccination_Event_Time_Interval)
days[np.isnan(days)] = 0
days[days < 0] = 0
minval = np.min(days[np.nonzero(days)])
maxval = np.max(days)
days = (days - minval)*0.8/(maxval-minval)+0.1
Towelve_months_data[:,indx]  = days.flatten()
indx = indx + 1

#vaccine adminestrated
Towelve_months_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

#Did the patient recieve a different vaccine
Towelve_months_data[:,indx] = np.zeros((len(df),))
indx = indx + 1

df.loc[df['12M Visit','Hospitalizations'].isna(), [['12M Visit','Hospitalizations']]]= 'MyNaN'
Towelve_months_data[:,indx]  = normalize_categorized(df['12M Visit','Hospitalizations'])
indx = indx + 1


df.loc[df['12M Visit','Documented COVID-19 infection'].isna(), [['12M Visit','Documented COVID-19 infection']]]= 'MyNaN'
Towelve_months_data[:,indx] = normalize_categorized(df['12M Visit','Documented COVID-19 infection'])
indx = indx + 1

df.loc[df['12M Visit','Rejection since last visit\xa0'].isna(), [['12M Visit','Rejection since last visit\xa0']]]= 'MyNaN'
Towelve_months_data[:,indx]  = normalize_categorized(df['12M Visit','Rejection since last visit\xa0'])
indx = indx + 1

df.loc[df['12M Visit', 'Any changes in immunosuppression medication since last visit?'].isna(), [['12M Visit', 'Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
Towelve_months_data[:,indx] = normalize_categorized(df['12M Visit', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1


headr = '12M Visit'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    Towelve_months_data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1
    
# headr = '12M Visit'
# column = 'More than one medication?'
# df[headr,column] = None
# Med = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
# tmp0 = np.zeros((len(df),len(Med)))
# for i in range(0,len(Med)): 
#     df.loc[df[headr,Med[i]].isna(), [[headr,Med[i]]]]= 'MyNaN'
#     tmp = df[headr, Med[i]]
#     tmp = tmp.replace('MyNaN', 0) 
#     tmp = tmp.replace('No', 0) 
#     tmp = tmp.replace('Yes', 1) 
#     tmp0[:,i] = tmp.values

# tmp1 = np.sum(tmp0, axis = 1)
# tmp1[tmp1 < 2] = 1
# tmp1[tmp1 >= 2] = 2
# Towelve_months_data[:,indx]  = tmp1
# indx = indx + 1

headr = '12M Visit'
Dose = ['Prednisone Dose (mg)',
        'Tacrolimus Dose (mg)',
        'Most recent Tac level\xa0',
        'Mycophenolate mofetil or mycophenolate sodium Dose (mg)']

Frequency = ['Prednisone Frequency',
             'Tacrolimus Frequency',
             'Mycophenolate mofetil or mycophenolate sodium Frequency']

for i in range(0,len(Dose)):
    df.loc[df[headr,Dose[i]].isna(), [[headr,Dose[i]]]]= 0
    Towelve_months_data[:,indx]  = normalize(df[headr,Dose[i]].astype(np.float32))
    indx = indx + 1
    
for i in range(0,len(Frequency)):
     df.loc[df[headr,Frequency[i]].isna(), [[headr,Frequency[i]]]]= 'MyNaN'
     tmp = df[headr, Frequency[i]]
     tmp = tmp.replace('MyNaN', 0) 
     tmp = tmp.replace('UNK', 0)
     tmp = tmp.replace('EOD', 1) #every othr day
     tmp = tmp.replace('Daily', 2) # once a day
     tmp = tmp.replace('BID', 3) #two times a day
     tmp = tmp.replace('TID',4) #three times a day
     Towelve_months_data[:,indx]  = tmp.values
     indx = indx + 1
     
print('visit 12M index is:' + str(indx))


#Masking where the 12Month antibody is not present
row, col = np.where(labels == 0)
row = row[col == 4]
Towelve_months_data[row,:]=0

permanent_data_12M = permanent_data.copy()
permanent_data_12M[row,:]=0
##############################################################################
# Saving data to a .npz file in long format
##############################################################################
data = np.zeros((len(df)*5,permanent_feature_length+visit_feature_size))
Mylabel = np.zeros((len(df)*5,1))
Myantibody = np.zeros((len(df)*5,1))

n = 0
for i in range(0,len(df)):
    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(visit_1_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2) , axis = 1)
    data[n,:] = tmp
    Mylabel[n]= labels[i,0]
    Myantibody[n] = Antibody[i,0]
    
    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(visit_2_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2) , axis = 1)
    data[n+1,:] = tmp
    Mylabel[n+1]= labels[i,1]
    Myantibody[n+1] = Antibody[i,1]
    
    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(visit_3_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2) , axis = 1)
    data[n+2,:] = tmp
    Mylabel[n+2]= labels[i,2]
    Myantibody[n+2] = Antibody[i,2]
    
    
    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(Six_months_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2) , axis = 1)
    data[n+3,:] = tmp
    Mylabel[n+3]= labels[i,3]
    Myantibody[n+3] = Antibody[i,3]


    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(Towelve_months_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2) , axis = 1)
    data[n+4,:] = tmp
    Mylabel[n+4]= labels[i,4]
    Myantibody[n+4] = Antibody[i,4]
    n = n + 5


path2save = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_Long_20240102.npz'
X = [data[0,:]]
y = [Mylabel[0]]
y_value = [Myantibody[0]]
for i in range(1,len(Mylabel)):
    X = np.append(X,[data[i,:]], axis=0)
    y = np.append(y,[Mylabel[i]])
    y_value = np.append(y_value,[Myantibody[i]])
    
# X = X/np.max(X,axis = 0)

antibody_values = y_value
y_value = np.log10(y_value)

 

# y_antibody = np.reshape(y_antibody,[len(df),5])

mymax = np.max(y_value)
y_value[np.isinf(y_value)] = mymax
mymin = np.min(y_value)



y_value = np.log10(Myantibody)




#np.min(y_antibody) = -0.3979400086720376
#np.max(y_antibody) = 6.056981036668113

y_value = (y_value-mymin)/(mymax-mymin)
y_value = y_value*0.8+0.1

#np.savez(path2save + fname, X = X, y = y, y_value = y_value, antibody_values = antibody_values)


##############################################################################
# Saving data to a .npz file in wide format
##############################################################################

data_wide = np.zeros((len(df),permanent_feature_length+visit_feature_size*5))
Mylabel = np.zeros((len(df),1))
myantibody = np.zeros((len(df),1))
n = 0
for i in range(0,len(df)):
    a1 = np.reshape(permanent_data[i,:],(1,permanent_feature_length))
    a2 = np.reshape(visit_1_data[i,:],(1,visit_feature_size))
    a3 = np.reshape(visit_2_data[i,:],(1,visit_feature_size))
    a4 = np.reshape(visit_3_data[i,:],(1,visit_feature_size))
    a5 = np.reshape(Six_months_data[i,:],(1,visit_feature_size))
    a6 = np.reshape(Towelve_months_data[i,:],(1,visit_feature_size))
    tmp = np.concatenate((a1,a2,a3,a4,a5,a6) , axis = 1)
    data_wide[i,:] = tmp
    Mylabel[i]= labels[i,4]
    myantibody[i] = Antibody[i,4]


path2save = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_wide_20240102.npz'
X = data_wide
y = Mylabel
y_value = myantibody
test = np.max(X,axis = 0)
test[test == 0]=1
X = X/test
y_value = y_value/np.max(y_value)*0.8+0.1

#np.savez(path2save + fname, X = X, y = y, y_value = y_value)

#############################################################
#1) have zero as a special character