# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:04:12 2024

@author: kasha
"""
immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']> 80]
non_immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']< 80]

ethinicity = ['white','south asian','chinese','black','filipino','latine Amrican','Arab',
              'southeast asian','west asian','korean','japanese','prefer to self describe','prefer not to answer']

for i in range(360,373):
    print('*********************' + ethinicity[i-360] + '*************************')
    print(non_immune[non_immune.columns[i]].value_counts())
    
    
from scipy.stats import mannwhitneyu

group1= normalize_categorized(immune['Characteristics','Sex'])

group2= normalize_categorized(non_immune['Characteristics','Sex'])

group1= immune.loc[immune['Characteristics','Age'].notna(),[['Characteristics','Age']]]

group2= non_immune.loc[non_immune['Characteristics','Age'].notna(),[['Characteristics','Age']]]

U1, p = mannwhitneyu(group1,group2)    
print(p)

Organ_Transplanted = ['Lung','Heart','Liver','Kidney','Pancreas']

indx = 4
for i in range(0,len(Organ_Transplanted)):
    column = 'Organ Transplanted (check all that apply) (choice=' + Organ_Transplanted[i] + ')'
    tmp = immune.loc[immune['Characteristics',column].notna(),[['Characteristics',column]]]
    group1 = normalize_categorized(tmp['Characteristics',column])
    tmp = non_immune.loc[non_immune['Characteristics',column].notna(),[['Characteristics',column]]]
    group2 = normalize_categorized(tmp['Characteristics',column])
    U1, p = mannwhitneyu(group1,group2)    
    print(Organ_Transplanted[i] + str(p))
    
    
    
vaccine_dose =  'Vaccine administered first dose'
vaccine_header = 'Characteristics'
df['first_dose_Pfizer'] = df[vaccine_header, vaccine_dose]
df.loc[df['first_dose_Pfizer'] == 'Pfizer', 'first_dose_Pfizer'] = 'Pfizer'
df.loc[df['first_dose_Pfizer'] == 'Moderna', 'first_dose_Pfizer'] = 'Other'
df.loc[df['first_dose_Pfizer'] == 'Astra Zeneca', 'first_dose_Pfizer'] = 'Other'

df['first_dose_Moderna'] = df[vaccine_header, vaccine_dose]
df.loc[df['first_dose_Moderna'] == 'Pfizer', 'first_dose_Moderna'] = 'Other'
df.loc[df['first_dose_Moderna'] == 'Moderna', 'first_dose_Moderna'] = 'Moderna'
df.loc[df['first_dose_Moderna'] == 'Astra Zeneca', 'first_dose_Moderna'] = 'Other'

df['first_dose_Astra_Zeneca'] = df[vaccine_header, vaccine_dose]
df.loc[df['first_dose_Astra_Zeneca'] == 'Pfizer', 'first_dose_Astra_Zeneca'] = 'Other'
df.loc[df['first_dose_Astra_Zeneca'] == 'Moderna', 'first_dose_Astra_Zeneca'] = 'Other'
df.loc[df['first_dose_Astra_Zeneca'] == 'Astra Zeneca', 'first_dose_Astra_Zeneca'] = 'Astra_Zeneca'

immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']> 80]
non_immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']< 80]

colums = ['first_dose_Pfizer','first_dose_Moderna','first_dose_Astra_Zeneca']
for col in colums: 
    group1= normalize_categorized(immune.loc[immune[col].notna(),col])

    group2= normalize_categorized(non_immune.loc[non_immune[col].notna(),col])

    U1, p = mannwhitneyu(group1,group2)    
    print(col + str(p))


vaccine_dose =  'Vaccine administered\xa0third dose'
vaccine_header = 'Third Dose Information'
df['third_dose_Pfizer'] = df[vaccine_header, vaccine_dose]
df.loc[df['third_dose_Pfizer'] == 'Pfizer', 'third_dose_Pfizer'] = 'Pfizer'
df.loc[df['third_dose_Pfizer'] == 'Moderna', 'third_dose_Pfizer'] = 'Other'
df.loc[df['third_dose_Pfizer'] == 'Astra Zeneca', 'third_dose_Pfizer'] = 'Other'
df.loc[df['third_dose_Pfizer'] == 'UNK', 'third_dose_Pfizer'] = 'Other'

df['third_dose_Moderna'] = df[vaccine_header, vaccine_dose]
df.loc[df['third_dose_Moderna'] == 'Pfizer', 'third_dose_Moderna'] = 'Other'
df.loc[df['third_dose_Moderna'] == 'Moderna', 'third_dose_Moderna'] = 'Moderna'
df.loc[df['third_dose_Moderna'] == 'Astra Zeneca', 'third_dose_Moderna'] = 'Other'
df.loc[df['third_dose_Moderna'] == 'UNK', 'third_dose_Moderna'] = 'Other'

df['third_dose_Astra_Zeneca'] =  df[vaccine_header, vaccine_dose]
df.loc[df['third_dose_Astra_Zeneca'] == 'Pfizer', 'third_dose_Astra_Zeneca'] = 'Other'
df.loc[df['third_dose_Astra_Zeneca'] == 'Moderna', 'third_dose_Astra_Zeneca'] = 'Other'
df.loc[df['third_dose_Astra_Zeneca'] == 'Astra Zeneca', 'third_dose_Astra_Zeneca'] = 'Astra_Zeneca'
df.loc[df['third_dose_Astra_Zeneca'] == 'UNK', 'third_dose_Astra_Zeneca'] = 'Other'

immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']> 80]
non_immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']< 80]

colums = ['fourth_dose_Pfizer','fourth_dose_Moderna','fourth_dose_Astra_Zeneca']
for col in colums: 
    group1= normalize_categorized(immune.loc[immune[col].notna(),col])

    group2= normalize_categorized(non_immune.loc[non_immune[col].notna(),col])

    U1, p = mannwhitneyu(group1,group2)    
    print(col + str(p))

#white
df.loc[df[df.columns[366]] == 'Checked', df.columns[360]]= 'Checked'
#Aisan
df.loc[df[df.columns[362]] == 'Checked', df.columns[361]]= 'Checked'
df.loc[df[df.columns[364]] == 'Checked', df.columns[361]]= 'Checked'
df.loc[df[df.columns[367]] == 'Checked', df.columns[361]]= 'Checked'
df.loc[df[df.columns[368]] == 'Checked', df.columns[361]]= 'Checked'
df.loc[df[df.columns[369]] == 'Checked', df.columns[361]]= 'Checked'
df.loc[df[df.columns[370]] == 'Checked', df.columns[361]]= 'Checked'

immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']> 80]
non_immune = df.loc[df['Antibody Information','12 month follow-up Antibody results (U/ml)']< 80]

colums = [df.columns[360],df.columns[363],df.columns[365],df.columns[361],df.columns[371]]
for col in colums: 
    group1= normalize_categorized(immune.loc[immune[col].notna(),col])

    group2= normalize_categorized(non_immune.loc[non_immune[col].notna(),col])

    U1, p = mannwhitneyu(group1,group2)    
    print(col[1][-20:] + str(p))
    
q75, q25 = np.percentile(immune['Characteristics','Age'], [75 ,25])
