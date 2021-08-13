# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:26:12 2021

Now used for two-to-one predictions but can be adapted for more predictors.
For 3 time series: uncomment 'X2'

@author: irisv
"""

import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import random
import datetime
import pandas as pd
import numpy as np
import copy


# Function for converting our dataset into the correct format
def window_data(df, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    Author: Johnny Wales https://github.com/walesdata/2Dconv_pub
    """
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i : (i + window), feature_col_number].values
        target = df.iloc[(i + window), target_col_number] # target = df.iloc[(i + window-1) zorgt voor overlap
        X.append(features)
        y.append(target) 
    return np.array(X), np.array(y).astype(np.float32).reshape(-1, 1)

def window_permutations(perm_data, window):
    """
    This function shapes the shuffled data for further processing.
    Perm_data : the shuffled time series, shuffledpredictor1 and shuffledpredictor2.
    Returns Reshaped permuted data in the form of a numpy array
    Adapted from Johnny Wales
    """
    Permuted_data = []
    for i in range(len(perm_data) - window - 1):
        features = perm_data.iloc[i : (i + window)].values
        Permuted_data.append(features)
    return np.array(Permuted_data)

def basic_conv2D(n_filters=10, fsize=4, window_size=5, n_features=2):
     new_model = keras.Sequential() 
     #new_model.add(tf.keras.layers.Conv2D(n_filters, (4,fsize), padding='same', data_format= 'channels_last', activation='relu', input_shape=(window_size, n_features, 1)))
     new_model.add(tf.keras.layers.Conv2D(n_filters, (4,fsize), padding='same', data_format= 'channels_last', activation='relu', input_shape=(window_size, n_features, 1)))
     new_model.add(tf.keras.layers.Flatten())
     new_model.add(tf.keras.layers.Dense(1000, activation='relu'))
     new_model.add(tf.keras.layers.Dense(100))
     new_model.add(tf.keras.layers.Dense(1))
     opt = keras.optimizers.Adam(learning_rate=0.005)
     new_model.compile(optimizer=opt, loss='mean_squared_error') 
     return new_model
 
def Average_cv(lst):
    return sum(lst) / len(lst)

def remove_last_element(arr, window_size):
    return arr[np.arange(arr.size - window_size-1)]

def findCONN(data, columns, window_size, fsize, significance, amount_of_predictors, predictor1, predictor2, target, number_of_run):
    
    num_of_splits=6
    tscv = TimeSeriesSplit(n_splits = num_of_splits)
    if len(data) == len(columns): data = np.transpose(data) # if rows== timeseries, then transpose such that cols become TSeries
    df_data = pd.DataFrame(data=data, columns=columns)
   
    # Copy predictors to get their shuffled version 
    pm_predictor1=copy.copy(df_data[predictor1])
    pm_predictor2=copy.copy(df_data[predictor2]) 
    random.shuffle(pm_predictor1)
    random.shuffle(pm_predictor2) 
    shuffled_predictor1= pm_predictor1
    shuffled_predictor2=pm_predictor2
     
    # Reshape      
    (X, y) = window_data(df_data, window_size, predictor1, target) 
    (X1, _) = window_data(df_data, window_size, predictor2, target) 
    shuffledX=window_permutations(shuffled_predictor1, window_size)
    shuffledX1=window_permutations(shuffled_predictor2, window_size)
  
    # Cross-validation for timeseries example https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
    r2_list=[]
    testdiff1_list=[]
    testdiff2_list=[]
    trainingloss_avg=[]
    diff_list=[]
    mse_testloss_shuffled_pred1=[]
    mse_testloss_shuffled_pred2=[]
    executiontime_list=[]
    i=1
    
    for train_index, test_index in tscv.split(df_data[predictor1]):

        if i==num_of_splits:
            test_index=remove_last_element(test_index, window_size)
        X_train, X_test = X[train_index], X[test_index]
        X1_train, X1_test = X1[train_index], X1[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        shuffledX_train, shuffledX_test=shuffledX[train_index], shuffledX[test_index]
        shuffledX1_train, shuffledX1_test =shuffledX1[train_index],shuffledX1[test_index] 
        
        # Reshape Traindata
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) # example shape (1194, 5, 1)
        X1_train = X1_train.reshape((X1_train.shape[0], X1_train.shape[1], 1))
        shuffledX_train=shuffledX_train.reshape((shuffledX_train.shape[0], shuffledX_train.shape[1], 1))
        shuffledX1_train=shuffledX1_train.reshape((shuffledX1_train.shape[0], shuffledX1_train.shape[1], 1))
        #X2_train = X2_train.reshape((X2_train.shape[0], X2_train.shape[1], 1))
        Xdata_train=np.concatenate((X_train, X1_train),axis=2)
        #Xdata_train=np.concatenate((X_train, X1_train,X2_train),axis=2)# 

        # Reshape Testdata
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) # (293, 5, 1)
        shuffledX_test=shuffledX_test.reshape((shuffledX_test.shape[0], shuffledX_test.shape[1], 1)) #  (293, 5, 1)
        X1_test = X1_test.reshape((X1_test.shape[0], X1_test.shape[1], 1))
        shuffledX1_test=shuffledX1_test.reshape((shuffledX1_test.shape[0], shuffledX1_test.shape[1], 1))
        #X2_test = X2_test.reshape((X2_test.shape[0], X2_test.shape[1], 1))
        Xdata_test=np.concatenate((X_test, X1_test),axis=2) # (299,6,5)
        #Xdata_test=np.concatenate((X_test, X1_test,X2_test),axis=2)
        shuffledXdata_test=np.concatenate((shuffledX_test, shuffledX1_test),axis=2) # (299,6,5)
        
        # X train and test wide
        data_train_wide = Xdata_train.reshape((Xdata_train.shape[0], Xdata_train.shape[1], Xdata_train.shape[2], 1))
        data_test_wide = Xdata_test.reshape((Xdata_test.shape[0], Xdata_test.shape[1], Xdata_test.shape[2], 1))
        permuted_testdata_wide=shuffledXdata_test.reshape((shuffledXdata_test.shape[0], shuffledXdata_test.shape[1], shuffledXdata_test.shape[2], 1))

      
        #fsize = width of the kernel 
        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1) # If you want to use early stopping inside the cv-folds
        m2 = basic_conv2D(n_filters=24, fsize=fsize, window_size=window_size, n_features=data_train_wide.shape[2])
        m2.summary() 
          
        # Train + timing, and fit 
        start_time = datetime.datetime.now()
        m2_hist = m2.fit(data_train_wide, y_train, epochs=12,validation_split=0, shuffle=False) 
        end_time = datetime.datetime.now()
        interval = (end_time - start_time)
        execution_time=interval.total_seconds()*1000
        print(i, 'training time for fold: %d mseconds' % int(execution_time))
        i=i+1
        
        #Print(m2_hist.history), get losses and accuracy and plot
        trainingloss=m2_hist.history["loss"]
        tr_loss_last=trainingloss[len(trainingloss)-1]

        diff=trainingloss[0]-tr_loss_last
        trainingloss_avg.append(trainingloss[0])
        diff_list.append(diff)
        #print("diff=trainingloss[0]-tr_loss_last:", diff)
        
        # plt.title("Trainloss - Conv2D Deep, w/ 1 target")
        # plt.plot(m2_hist.history["loss"],label="Train")
        # plt.legend(["loss"])
        # plt.show() 
    
        #Evaluate on testset, get MSE from keras, and from scikitlearn (double check) + evaluate with shuffled testdata
        eval_output_y=m2.evaluate(data_test_wide, y_test, verbose=0) 
 
        # Get predictions and R2 score
        predictions = m2.predict(data_test_wide)
        mse_y=mean_squared_error(y_test, predictions[:,0]) 
        r2_y = r2_score(y_test, predictions) 
        r2_list.append(r2_y)
        
        
        for shuffling in range(0,2):
            if shuffling==0 : 
                # using shuffled predictor1
                shuffledXdata_test=np.concatenate((shuffledX_test, X1_test),axis=2)
                permuted_testdata_wide=shuffledXdata_test.reshape((shuffledXdata_test.shape[0], shuffledXdata_test.shape[1], shuffledXdata_test.shape[2], 1))
                predictions_with_shuffled_testdata=m2.predict(permuted_testdata_wide) # loop
                mse_testloss_shuffled_first_predictor=mean_squared_error(y_test, predictions_with_shuffled_testdata[:,0])
                mse_testloss_shuffled_pred1.append(mse_testloss_shuffled_first_predictor)
                testdiff1=trainingloss[0]-mse_testloss_shuffled_first_predictor
                testdiff1_list.append(testdiff1)
            if shuffling==1: 
                # using shuffled predictor2
                shuffledXdata_test=np.concatenate((X_test, shuffledX1_test), axis=2)
                permuted_testdata_wide=shuffledXdata_test.reshape((shuffledXdata_test.shape[0], shuffledXdata_test.shape[1], shuffledXdata_test.shape[2], 1))
                predictions_with_shuffled_testdata=m2.predict(permuted_testdata_wide) # loop
                mse_testloss_shuffled_second_predictor=mean_squared_error(y_test, predictions_with_shuffled_testdata[:,0])
                mse_testloss_shuffled_pred2.append(mse_testloss_shuffled_second_predictor)
                testdiff2=trainingloss[0]-mse_testloss_shuffled_second_predictor
                testdiff2_list.append(testdiff2)
        
        executiontime_list.append(execution_time)

    total_time_for_one_run=sum(executiontime_list)
    #print("avg-cv testdiff1:", testdiff1, " and avg newdiff*sig: ",  diff*significance)
    #print("avg-cv testdiff2:", testdiff2, " and avg newdiff*sig: ", diff*significance)
            
    #Show target and its predictions in a new dataframe (take them from last fold)
    # conv_acc_df = pd.DataFrame()
    # conv_acc_df['Actual'] = y_test[:,0]
    # conv_acc_df['Predict'] = predictions[:,0] # fill df with predictions: all rows but only the first col (second col is 'dtype float32)
    # conv_acc_df.head(10)
    # Visualize the fit #######################
    # plt.title("Target vs. Prediction, shape: ")
    # plt.plot(conv_acc_df[160:210])
    # plt.show()


    return r2_list, total_time_for_one_run, testdiff1_list, testdiff2_list, diff_list

