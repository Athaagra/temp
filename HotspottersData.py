#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kel
"""
import os
import math
import csv
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import string
from tensorflow.keras.utils import to_categorical
from math import sin, cos, sqrt, atan2, radians
import datetime
from dateutil.parser import parse

# Set ggplot styles and update Matplotlib with them.
ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': 'EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': '1.2',
    'xtick.color': '555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)
dz = np.repeat(0.812345, 1000, axis=0)
dz = [0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345, 0.812345]
os.chdir('/home/kel/Desktop/MasterThesis/TrajectoryData')
def normalize(a):
    amin, amax = min(a), max(a)
    for i,val in enumerate(a):
        a[i] = 2*(val-amin)/(amax-amin)-1
    return a

def retrieve_data(AndroidData):
    DataHeaders = list(AndroidData.columns.values)
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    dataCl=[]
    with open('db_backup_2021-10-27_1550.csv') as fp:
        line = str(fp.readlines())
        x=re.split('[a-z]+',line)
        for l in x:
            if len(l)>10 and len(l)<30:
                lp=re.sub('[^a-zA-Z0-9\n\.\s+]+', ' ',l)
                lp=re.sub(r"\s+", "", lp, flags=re.UNICODE)
                all_uppercase = lp.isupper()
                if all_uppercase==True:
                    dataCl.append(re.sub(r"\s+", "", lp, flags=re.UNICODE))
                count_p=0
                for i in lp:
                    if i in string.punctuation:
                        count_p+=1
                #cpunct=[count_punctuation=count_punctuation+1 if i in string.punctuation for i in lp]
                if count_p >= 5 and count_p<10:
                    lp=datetime.datetime.strptime(lp, '%Y.%m.%d.%H.%M.%S')
                    print(lp)
                    dataCl.append(lp)
                else:
                    dataCl.append(lp)
            elif len(l) >4000:
                lo=re.sub('[^a-zA-Z0-9\n\.\s+]+', ' ',l)
                lo=[float(q) for q in lo.split()]
                lo_array=np.array(lo)
                dataCl.append(lo)         
    print(AndroidData.head())
    
    plt.style.use('ggplot')
    #figure
    plt.figure(figsize=(10,8))
    #both distribution in one diagram
    plt.plot(dataCl[0], 'r',label='x')
    plt.plot(dataCl[1],'b',label='y')
    plt.plot(dz, 'g',label='z')
    #title of diagram
    plt.title("Rotation plot")
    #label y title
    plt.ylabel("Degrees")
    #label x title
    plt.xlabel("time")
    #save the figure
    plt.savefig("IndegreeAndOutdegreeSmall.png")
    #display the figure
    plt.legend(loc="upper right")
    plt.show()
    Sensors = []
    for i in dataCl:
        if type(i)==list:
            Sensors.append(i)

    number_list=[0,1]
    import random
    G = random.choice(number_list)
    SensorsData=[]
    for x in range(2,len(Sensors)):
        SensorsData.append([Sensors[x-2],Sensors[x-1],Sensors[x],np.tile(G,len(Sensors[x]))])
    return SensorsData

AndroidDataF = pd.read_csv('db_backup_2021-10-27_1550.csv', delimiter = ',',quoting=csv.QUOTE_NONE)
AndroidDataE = pd.read_csv('db_backup_2021-10-27_1558.csv', delimiter = ',',quoting=csv.QUOTE_NONE)
AndroidDataS = pd.read_csv('db_backup_2021-10-27_1615.csv', delimiter = ',',quoting=csv.QUOTE_NONE)
DataF=retrieve_data(AndroidDataF)
DataE=retrieve_data(AndroidDataE)
DataS=retrieve_data(AndroidDataS)
#from collections import Counter
#a = dict(Counter(dataCl))
#NGSIMGr=NGSIM.groupby(['Vehicle_ID','Location'])
#NGSIMsor=[]
#for key, item in NGSIMGr:
   #print(item)
#   item.sort_values("Global_Time", axis = 0, ascending = True, 
 #                inplace = True, na_position ='last')
#   item=item[0:12]
#   NGSIMsor.append(item)

#NGSIMlst = pd.concat(NGSIMsor)
#NGSIMlst = NGSIMlst.dropna()
#Time = np.array(NGSIMlst['Global_Time'])
#Velocity = np.array(NGSIMlst['v_Vel'])
#Movement = np.array(NGSIMlst['Movement'])
#longitude = np.array(NGSIMlst['Local_X'])
#latitude = np.array(NGSIMlst['Local_Y'])
#Glongitude = np.array(NGSIMlst['Global_X'])
#Glatitude = np.array(NGSIMlst['Global_Y'])
#vehicle = np.array(NGSIMlst['Vehicle_ID'])
#R=6378.1
#Velx=[]
#Vely=[]
#for i in range(12,len(NGSIMlst)):
#    Vy=radians(longitude[i-11]-longitude[i-10]) * R / Time[i-11]-Time[i-10]
#    Vx=radians(latitude[i-11]-latitude[i-10]) * R / Time[i-11]-Time[i-10]
#    Velx.append(Vx)
#    Vely.append(Vy)
#VeltoTar=[]
#Veltarg=[]
#LattoTar=[]
#Lattarg=[]
#LontoTar=[]
#Lontarg=[]
#TCCp=[]
#for q in range(12, len(Vely)):
#    targety = Vely[q]
#    xtar = latitude[q]
#    Dxp = latitude[q-11:q-1]-xtar
#    Duyp = Vely[q-11:q-1]-targety
#    ytar = longitude[q]
#    Dyp = longitude[q-11:q-1]-ytar
#    TTCpx = Dyp/Duyp
#    VeltoTar.append(Duyp)
#    Veltarg.append(targety)
#    LattoTar.append(Dxp)
#    Lattarg.append(xtar)
#    LontoTar.append(Dyp)
#    Lontarg.append(ytar)
#    TCCp.append(TTCpx)

#def features(x,y):
#    feature_set=[]
#    labels=[]
#    for t in range(12, len(x)):
#        feature_set.append(x[t-11:t-1])
#        labels.append(y[t])
#    return(feature_set,labels)

def features(SensorsD):
    feature_set=[]
    for i in range(0, len(SensorsD)):
        for k in range(0, len(SensorsD[i][3])):
            feature_set.append([SensorsD[i][0][k],SensorsD[i][1][k],SensorsD[i][2][k],SensorsD[i][3][k]])
    return np.array(feature_set)
#Velx = normalize(Velx)
feature_setE= features(DataE)
feature_setS= features(DataS)
feature_setF= features(DataF)
sets=np.vstack((feature_setE,feature_setF))
feature_set=np.vstack((sets,feature_setS))
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler.fit_transform()
dataset = feature_set
labels = feature_set[:,3]
#labels = normalize(labels)
#labels_scaled=scaler.fit_transform(labels)
# create training and testing vars
lastLayer=len(np.unique(labels)+1)
trainX, testX, trainY, testY = train_test_split(dataset, labels, test_size=0.2)
# split into train and test sets
# reshape input to be [samples, time steps, features]
n_steps=1
#trainY = keras.utils.to_categorical(trainY)
#testY = keras.utils.to_categorical(testY)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
trainY = np.asarray(trainY)
# create and fit the LSTM network
n_features = 10
model = Sequential()
model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_features, n_steps)))
model.add(LSTM(256, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)
print(history.history, file=open('LSTM-HistorySensor.txt', 'w'))
# make predictions
prtrain = model.predict(trainX)
prtest = model.predict(testX)
# invert predictions
#trainY = np.argmax(y_train, axis=1)
#tr = cat*prtrainlo
#trainPredict=tr.sum(axis=1)
#testY = np.argmax(y_test, axis=1)
#te = cat*prtest
#testPredict=te.sum(axis=1)
#trainPredict = scaler.inverse_transform(prtrain)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(prtest)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainY=np.array(trainY)
testY=np.array(testY)
trainScore = math.sqrt(mean_squared_error(trainY, prtrain[:,0]))
print('Train Score: %.2f RMSE' % (trainScore), file=open('LSTM-TrainSet Score-longt-Velocit.txt', 'w'))
testScore = math.sqrt(mean_squared_error(testY, prtest[:,0]))
print('Test Score: %.2f RMSE' % (testScore), file=open('LSTM-TestSet Score-longt-Velocit.txt', 'w'))
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(testY, prtest[:,0])

loss=[0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595, 0.6667627692222595]
losL=np.arange(0,len(loss))

import matplotlib.pyplot as plt
plt.figure(figsize=(18, 18))
plt.plot(loss, color=['r'])
#plt.grid(True,which="both",ls="--",c='gray') 
L=plt.legend()
L.get_texts()[0].set_text('Distance')
#L.get_texts()[1].set_text('Speed')
plt.show()

plt.figure(figsize=(18,18))
# plot  a figure a size 10,8 with blue color and the symbol ^
plt.loglog(loss, linestyle='--', color='r')
# label of diagram
plt.title("Loss error plot")
# label of y
plt.ylabel("Loss")
# label of x
plt.xlabel("iterations")
# save the figure
plt.savefig("Loss_Lineplot.png")
# display the plot
plt.show()
