# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 2021

Now used for one-to-one predictions but can be adapted for more predictors (line 91).
See also amount_of_used_predictors in runCNND_Basic.py, which defines the amount of shuffled predictors that need to be created

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
        target = df.iloc[(i + window), target_col_number] 
        X.append(features)
        y.append(target) 
    return np.array(X), np.array(y).astype(np.float32).reshape(-1, 1)

def window_permutations(perm_data, window):
    """
    This function shapes the shuffled data for further processing.
    Perm_data : the shuffled time series.
    Returns Reshaped permuted data in the form of a numpy array.
    Adapted from Johnny Wales
    """
    Permuted_data = []
    for i in range(len(perm_data) - window - 1):
        features = perm_data.iloc[i : (i + window)].values
        Permuted_data.append(features)
    return np.array(Permuted_data)

def basic_conv2D(n_filters=10, fsize=4, window_size=5, n_features=2):
     new_model = keras.Sequential() 
     new_model.add(tf.keras.layers.Conv2D(n_filters, (4,fsize), padding='same', data_format= 'channels_last', activation='relu', input_shape=(window_size, n_features, 1)))
     #new_model.add(tf.keras.layers.Conv2D(n_filters, (4,fsize), padding='same', data_format= 'channels_last', activation='relu', input_shape=(window_size, n_features, 1)))
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

def findCONN(data, columns, window_size, fsize, significance, amount_of_predictors, new_predictor, predictorlist, targetseries):
    
    num_of_splits=6
    tscv = TimeSeriesSplit(n_splits = num_of_splits)
    
    if len(data) == len(columns): data = np.transpose(data) # if rows== timeseries, then transpose such that cols become TSeries
    df_data = pd.DataFrame(data=data, columns=columns)
    # Print the first few rows of the dataset
    #print("First rows of data, predictorlist: ", df_data.head(2))  
 
    # Depending on the amount of predictors: create shuffled predictors
    pm_predictor_list=[]
    shuffled_predictorlist=[]
    for i in range (0,amount_of_predictors): 
        pm_predictor_list.append(copy.copy(df_data[predictorlist[i]])) 
        random.shuffle(pm_predictor_list[i])
        shuffled_predictorlist.append(pm_predictor_list[i])
    len(shuffled_predictorlist)
    
     
    # Depending on the amount of predictors: create windowed predictor, do the same for shuffled ones.  
    if len(pm_predictor_list)==1:     
        (X, y) = window_data(df_data, window_size, predictorlist[new_predictor], targetseries)
        shuffledX=window_permutations(shuffled_predictorlist[0], window_size)
        scenario=1
    elif len(pm_predictor_list)==2:
        (X, y) = window_data(df_data, window_size, predictorlist[0], targetseries)    
        (X1, _) = window_data(df_data, window_size, predictorlist[1], 1) 
        shuffledX=window_permutations(shuffled_predictorlist[0], window_size)
        shuffledX1=window_permutations(shuffled_predictorlist[1], window_size)
        scenario=2
    elif len(pm_predictor_list)==3:   
        (X, y) = window_data(df_data, window_size, predictorlist[0], targetseries)    
        (X1, _) = window_data(df_data, window_size, predictorlist[1], 1) 
        (X2,_)=window_data(df_data, window_size, predictorlist[2], 1) 
        shuffledX=window_permutations(shuffled_predictorlist[0], window_size)
        shuffledX1=window_permutations(shuffled_predictorlist[1], window_size)
        shuffledX2=window_permutations(shuffled_predictorlist[2], window_size)
        scenario=3
    else:
        print("Please use 3 predictors or less. The script is not adapted yet for more")
        
    # Example Cross-validation for timeseries https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
    r2_list=[]
    testdiff_list=[]
    trainingloss_avg=[]
    diff_list=[]
    mse_testloss_shuffled_avg=[]
    executiontime_list=[]
    
    i=1
    
    for train_index, test_index in tscv.split(df_data[0]): 
        if i==num_of_splits:
            test_index=remove_last_element(test_index, window_size)
        #print('test index after removal: ', test_index)      
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        shuffledX_train, shuffledX_test=shuffledX[train_index], shuffledX[test_index]
        shuffledX_train=shuffledX_train.reshape((shuffledX_train.shape[0], shuffledX_train.shape[1], 1))
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) #e.g. (1194, 5, 1)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) # (293, 5, 1)
        shuffledX_test=shuffledX_test.reshape((shuffledX_test.shape[0], shuffledX_test.shape[1], 1)) #  (293, 5, 1)
        
        def create_shaped_variables(var, shuffled_var,train_index, test_index):
            var_train, var_test=var[train_index],var[test_index]
            shuffled_var_train, shuffled_var_test=shuffled_var[train_index], shuffled_var[test_index]
            var_train = var_train.reshape((var_train.shape[0], var_train.shape[1], 1))
            shuffled_var_train = shuffled_var_train.reshape((shuffled_var_train.shape[0], shuffled_var_train.shape[1], 1))
            
            var_test= var_test.reshape((var_test.shape[0], var_test.shape[1], 1))
            shuffled_var_test=shuffled_var_test.reshape((shuffled_var_test.shape[0], shuffled_var_test.shape[1], 1))
            return var_train, shuffled_var_train, var_test, shuffled_var_test
        
        # Create train and test sets for more than 1 predictor
        if scenario==2:  
            X1_train, shuffled_X1_train, X1_test, shuffled_X1_test=create_shaped_variables(X1, shuffledX1, train_index, test_index)

        if scenario==3:
            X1_train, shuffled_X1_train, X1_test, shuffled_X1_test=create_shaped_variables(X1, shuffledX1, train_index, test_index)
            X2_train, shuffled_X2_train, X2_test, shuffled_X2_test=create_shaped_variables(X2, shuffledX2, train_index, test_index)

        # if one predictor variable:
        if scenario==1:
            Xdata_train=X_train 
            Xdata_test=X_test

            shuffledXdata_test=shuffledX_test
        elif scenario==2: # if 2 predictors
            Xdata_train=np.concatenate((X_train, X1_train),axis=2)
            Xdata_test=np.concatenate((X_test, X1_test),axis=2) # (299,6,5)

            shuffledXdata_test=np.concatenate((shuffledX_test,shuffled_X1_test),axis=2)
        else: # if 3 predictors 
            Xdata_train=np.concatenate((X_train, X1_train, X2_train),axis=2)
            Xdata_test=np.concatenate((X_test, X1_test, X2_test),axis=2)
            shuffledXdata_test=np.concatenate((shuffledX_test,shuffled_X1_test, shuffled_X2_test),axis=2)
        # X wide (Reshaped data)
        data_train_wide = Xdata_train.reshape((Xdata_train.shape[0], Xdata_train.shape[1], Xdata_train.shape[2], 1))             
        data_test_wide = Xdata_test.reshape((Xdata_test.shape[0], Xdata_test.shape[1], Xdata_test.shape[2], 1))
        permuted_testdata_wide=shuffledXdata_test.reshape((shuffledXdata_test.shape[0], shuffledXdata_test.shape[1], shuffledXdata_test.shape[2], 1))
    
        #fsize = width of kernel
        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1) # use early stopping inside cv if wanted
        m2 = basic_conv2D(n_filters=24, fsize=fsize, window_size=window_size, n_features=data_train_wide.shape[2])
        m2.summary() # https://discuss.pytorch.org/t/forward-block-in-inception-network-for-creating-basicconv2d/77496/2
          
        # Train + timing, and fit 
        start_time = datetime.datetime.now()
        m2_hist = m2.fit(data_train_wide, y_train, epochs=12, validation_split=0, shuffle=False) #callbacks=[callback],validation_split=0.10
        end_time = datetime.datetime.now()
        interval = (end_time - start_time)
        execution_time=interval.total_seconds()*1000
        print(i, 'th fold. training time for fold: %d mseconds' % int(execution_time))
        i=i+1
        
        #Print(m2_hist.history), get losses and accuracy and plot
        trainingloss=m2_hist.history["loss"]
        tr_loss_last=trainingloss[len(trainingloss)-1]
        diff=trainingloss[0]-tr_loss_last
        
        # plt.title("Trainloss - Conv2D, w/ 1 target ")
        # plt.plot(m2_hist.history["loss"],label="Train")
        # plt.legend(["loss"])
        # plt.show() 
    
        #Evaluate on testset, get MSE from keras, and from scikitlearn (double check) 
        eval_output_y=m2.evaluate(data_test_wide, y_test, verbose=0) 
 
        # Get R2 score, Permutation Importance https://medium.com/@vivek_skywalker/permutation-importance-a1df5010fa99
        predictions = m2.predict(data_test_wide)
        # get predictions with shuffled data
        predictions_with_shuffled_testdata=m2.predict(permuted_testdata_wide) # all predictors shuffled
       
        mse_y=mean_squared_error(y_test, predictions[:,0])
        r2_y = r2_score(y_test, predictions)
        r2_list.append(r2_y)

        # Fetch testloss, training differences and execution times
        mse_testloss_shuffled=mean_squared_error(y_test, predictions_with_shuffled_testdata[:,0]) 
        mse_testloss_shuffled_avg.append(mse_testloss_shuffled)
        trainingloss_avg.append(trainingloss[0])
        diff_list.append(diff)
        executiontime_list.append(execution_time)
        
        # Test difference
        testdiff=trainingloss[0]-mse_testloss_shuffled 
        testdiff_list.append(testdiff)
        
  
    #execution_time=Average_cv(executiontime_list)
    total_time_for_one_run=sum(executiontime_list) 
           
    #Show target and its predictions in a new dataframe 
    # conv_acc_df = pd.DataFrame()
    # conv_acc_df['Actual'] = y_test[:,0]
    # conv_acc_df['Predict'] = predictions[:,0] # fill df with predictions: all rows but only the first col (second col is 'dtype float32)
    # conv_acc_df.head(10)
    # # Visualize the fit
    # plt.title("Target vs. Prediction, shape: ")
    # plt.plot(conv_acc_df[160:210])
    # plt.show()


    return r2_list, total_time_for_one_run, testdiff_list, diff_list, eval_output_y, mse_y



















