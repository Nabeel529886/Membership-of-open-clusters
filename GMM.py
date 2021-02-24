"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:14:18 2020

@author: kkhushlim
"""
"""The code computes the likely cluster members using the Gaussian Mixture
Model Clustering Algorithm. Here we are using a two component Gaussian model
since we have two clusters in our dataset"""
#import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt


#importing the data file
sample = pd.read_csv("NGC2112.csv")
print("Shape of Dataset: ", sample.shape)

#Drop rows that have no values
####################
sample.dropna(inplace = True)
print('After dropping rows that contain Nan: ', sample.shape)
sample.to_csv('sample.csv')
####################

#selecting desired coloumns only
####################
data = sample[['ra', 'dec', 'parallax','pmra', 'pmdec']]
print(data.head(2))
####################

#Normalizing the data
def normalize(dataset):
    dataNorm=(dataset-dataset.median())/dataset.std()
    #dataNorm["id"]=dataset["id"]
    return dataNorm

data=normalize(data)
#print(data.sample(5))
####################

#fitting the Gaussian model to the sampel data
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data)
#print(gmm.means_)
#print(gmm.covariances_)
labels = gmm.predict(data)
mem_prob = gmm.predict_proba(data)
#prob = mem_prob.tolist()
prob= [comp[1] for comp in mem_prob]
frame =pd.DataFrame(sample)
frame['cluster'] = labels
frame['prob']= prob
frame.to_csv('output.csv')
###################

#sorting cluster members and field stars to two different files from the output file
df = pd.read_csv("output.csv")
df= df[["source_id", "ra", "dec", "parallax", "pmra", "pmdec", "bp_rp", "phot_g_mean_mag", "cluster", "prob"]]
print("Shape of Dataset: ", df.shape)
mem_stars = df.loc[df['cluster'] == 1]
mem_stars.to_csv("cluster.csv")
field_stars = df.loc[df['cluster'] == 0]
field_stars.to_csv("field.csv")

###################
plt.figure(figsize=(3,3))
color=['blue', 'red']
for i in range(0,2):
    data = frame[frame["cluster"]==i]
    #plt.scatter(data["ra"],data["dec"], s =2**2, c=color[i])
    #plt.scatter(data["pmra"],data["pmdec"], s =1**2, c=color[i])
    plt.scatter(data["bp_rp"],data["phot_g_mean_mag"], s = 0.5**2, c=color[i])
    #plt.scatter(data['phot_g_mean_mag'], data["prob"], s= 2**2, c=color[i])
#plt.xlim(-15, 15)
#plt.ylim(-15, 15)
#plt.grid(False)
plt.gca().invert_yaxis()
#plt.legend()
#plt.savefig('CMD_NGC188.png')
plt.show();
###################
