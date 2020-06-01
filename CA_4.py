# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:09:54 2020

@author: quinn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('World_Happiness_2019.csv')



# reading the data in
dataset = pd.read_csv('World_Happiness_2019.csv')
# converting to dataframe
df = pd.DataFrame(data=dataset)
#display all columns from dataset 
pd.set_option('display.max_columns', None)

# examining data 

# display top 5 rows from dataset
print (df.head())
 # display the size of the dataset - 156 x 9
print (df.shape)
# display dataframe info - datatypes missing values etc.   
print (df.info()) 
# display basic info on each column (rows, mean, std, etc.)
print (df.describe())
# display all column names
print (df.columns) 

#check  for missing data
pd.isna(dataset)
print('Nan' , dataset.isna().sum(), sep='\n')



#Heat map
corrmat = df.corr()
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, square=True)


#top 10 countries by attribute
fig1, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(12,8))

sns.barplot(x='GDP per capita',y='Country or region',
            data=df.nlargest(10,'GDP per capita'),ax=axes[0,0],
            palette="Blues_d")

sns.barplot(x='Social support' ,y='Country or region',
            data=df.nlargest(10,'Social support'),ax=axes[0,1],
            palette="YlGn")

sns.barplot(x='Healthy life expectancy' ,y='Country or region',
            data=df.nlargest(10,'Healthy life expectancy'),ax=axes[1,0],
            palette='OrRd')

sns.barplot(x='Freedom to make life choices' ,y='Country or region',
            data=df.nlargest(10,'Freedom to make life choices'),ax=axes[1,1],
            palette='YlOrBr')


fig2, axes = plt.subplots(nrows=1, ncols=2,constrained_layout=True,figsize=(10,4))

sns.barplot(x='Generosity' ,y='Country or region',
            data=df.nlargest(10,'Generosity'),ax=axes[0],palette='Spectral')
sns.barplot(x='Perceptions of corruption' ,y='Country or region'
            ,data=df.nlargest(10,'Perceptions of corruption'),ax=axes[1],
            palette='RdYlGn')

#Load in  top 10
d= df[(df['Country or region'].isin(['Finland','Denmark','Norway',
       'Iceland','Netherlands','Switzerland','Sweden','New Zealand','Canada','Austria','Ireland']))]
print (d)

# Plot top 10 in stacked bar
ax = d.plot(y="Social support", x="Country or region", kind="bar",color='C3')
d.plot(y="GDP per capita", x="Country or region", kind="bar", ax=ax, color="C1")
d.plot(y="Healthy life expectancy", x="Country or region", kind="bar", ax=ax, color="C2")

plt.show()


