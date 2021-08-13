# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7  2021

Script to run all possible combinations of two-to-one predictions.
Datasets need to be defined, no ground truth is provided.
Output = score matrices with significant R2-scores, and runtime (how to export time can be defined below in the file)
Extra outputs: R2 scores, significance weights, standard deviation of R2-scores, standard deviation of runtime if defined for export
Parameters to set: amount of datasets to use (num_datasets), which datasets to use (alldata, in the case others are also defined in the file), how many runs of the same dataset to average over (average, default=5)
Other parameters that have a default value but can be changed include width_of_kernel (default=4), learning_rate (0.005), significance=0.9998, and
window_size (default=5) i.e. Window that should be considered to preprocess (reshape) the timeseries for a conv2d
@author: irisv
"""

import pandas as pd
import numpy as np
import csv
from CNNd_multivar import findCONN


# Where should the output be saved and under which name?
name="filename_to_fill_in" 
path= "C:/Users/irisv/Desktop/path/end/"

# Datafiles to load
# dataset1=pd.read_csv('DataFarSuperficial1500_1.csv')
# dataset2=pd.read_csv('DataFarSuperficial1500_2.csv')
# dataset3=pd.read_csv('DataFarSuperficial1500_3.csv')
# dataset4=pd.read_csv('DataFarSuperficial1500_4.csv')
# dataset5=pd.read_csv('DataFarSuperficial1500_5.csv')


dataset1=pd.read_csv('DataFarDeep3T1500_june_1.csv') 
dataset2=pd.read_csv('DataFarDeep3T1500_june_2.csv')
dataset3=pd.read_csv('DataFarDeep3T1500_june_3.csv')
dataset4=pd.read_csv('DataFarDeep3T1500_june_4.csv')
dataset5=pd.read_csv('DataFarDeep3T1500_june_5.csv')




# Amount of datasets to use
num_datasets=5
alldata=[dataset1, dataset2, dataset3, dataset4, dataset5]
# Amount of runs to average over, default=5
average=range(0,5)

## TESTING SCRIPT with 2 sets
# num_datasets=2
# alldata=[dataset1,dataset2]


width_of_kernel=4  
learning_rate=0.005 
significance=0.9998 
window_size=5 # Window that should be considered to preprocess (reshape) the timeseries for a conv2d
seed=1000 # seed  

#How many Timeseries are used?
amount_of_timeseries=3
amount_of_used_predictors=2
columns=range(0,amount_of_timeseries)
predictorlist=[0,1,2]

time_per_dataset=[]
time=0
totaltime=0
all_matrices=[]
all_m1=[]
all_times=[]

all_r2scores=[]
all_r2_sd=[]
all_r2_averages=[]
all_significanceweights1=[]
all_significanceweights2=[]


for num in range(0,num_datasets): 
    data=alldata[num]
    score_matrix = np.zeros((data.shape[0], data.shape[0]))
    matrix1 = np.zeros((3, 3, 3))
    matrix2 = np.zeros((3, 3, 3))
    time_per_dataset=0

    for k in range(0,3): # targetseries
        
        r2scores=[]
        r2_sds=[]
        r2_averages=[]
        significanceweights1=[]
        significanceweights2=[]
    
        for j in range(0,len(predictorlist)):
        
                
            if j==0:
                predictor1=0  
                predictor2=1
                targetseries=k 
            elif j==1:
                predictor1=1  
                predictor2=2
                targetseries=k
            else:
               predictor1=0  
               predictor2=2
               targetseries=k    
            
            allscores=[]
            testdiff_list=[]
            testdiff1_list=[]
            testdiff2_list=[]
            trainingdiff_list=[]
            significancelist=[]
            times_avg5_runs=0
  
            def Average(lst):
                return sum(lst) / len(lst)
            
            for i in average: 
            
                r2_scorelist, runtime, test1_differences, test2_differences, training_differences=findCONN(data, columns, window_size, width_of_kernel, significance, amount_of_used_predictors, predictor1, predictor2, targetseries, i)
                allscores.append(np.array(r2_scorelist)) 
                testdiff1_list.append(test1_differences)
                testdiff2_list.append(test2_differences)
                trainingdiff_list.append(training_differences)
                totaltime=totaltime+runtime
            times_avg5_runs=times_avg5_runs + totaltime
            time=time+times_avg5_runs
            new_testdiff1=np.mean(testdiff1_list)
            new_testdiff2=np.mean(testdiff2_list)
            new_diff=np.mean(trainingdiff_list)
            significance_weight1=new_testdiff1/new_diff
            significance_weight2=new_testdiff2/new_diff
            average_r2 = np.mean(allscores) # take mean of the whole list
            r2_sd=np.std(allscores)
            score_matrix[j][k]=round(average_r2,4)
            
            if new_testdiff1>(new_diff*significance):        
                    print("Significance= ", significance_weight1, "R2_score average =", round(average_r2, 4))
                    matrix1[j][predictor1][k]=0 
                    print('Connectivity for predictor ',predictor1,' not significant') 
            else:
                if significance_weight1>0.7 or average_r2<0: # threshold for significance weights, the lower, the more restrictive (less positives)
                    print("Significance =", round(significance_weight1, 4))
                    print('Connectivity for predictor ',predictor1, ' and target ', k, ' NOT significant')
                else:
                    matrix1[j][predictor1][k]=round(significance_weight1, 4)
                    print("R2_score average =", round(average_r2, 4), "Total traintime= ", totaltime)
                    print("Significance =", round(significance_weight1, 4))
                    print('Connectivity for predictor ',predictor1, ' and target ', k, ' significant')
                
            if new_testdiff2>(new_diff*significance):        
                    matrix1[j][predictor2][k]=0 
                    print("sig= ", significance_weight2, "R2_score average =", round(average_r2, 4))
                    print('Connectivity for predictor ',predictor2,' not significant') 
            else:
                if significance_weight2>0.7 or average_r2<0: 
                    print("Significance =", round(significance_weight2, 4))
                    print('Connectivity for predictor ',predictor2, ' and target ', k, ' NOT significant')
                else:#%was 0.5
                    matrix1[j][predictor2][k]=round(significance_weight2, 4)
                    print("R2_score average =", round(average_r2, 4), "Total traintime= ", totaltime)
                    print("Significance =", round(significance_weight2, 4))
                    print('Connectivity for predictor ',predictor2, ' and target ', k, ' significant')
        

            r2_averages.append(average_r2)
            r2_sds.append(r2_sd)
            significanceweights1.append(significance_weight1)
            significanceweights2.append(significance_weight2)

            

        all_r2_averages.append(r2_averages) # average to report over five datasets
        all_r2_sd.append(r2_sds) # average to report over five datasets
        all_significanceweights1.append(significanceweights1)
        all_significanceweights2.append(significanceweights2)
  

    # score matrix for one dataset
    print(score_matrix)
    print(matrix1)
    time_per_dataset=times_avg5_runs
    print(time_per_dataset)
    
    # get all matrices and times for the datasets
    all_matrices.append(score_matrix)
    all_m1.append(matrix1)
    all_times.append(time_per_dataset)
    

#print(time, time_per_dataset[0])
TIME1= all_times[0]
TIME2= all_times[1]-TIME1
TIME3=all_times[2]-TIME2
TIME4=all_times[3]-TIME3
TIME5=all_times[4]-TIME4
average_TIME_per_dataset= (TIME1+TIME2+TIME3+TIME4+TIME5)/5
TIME_IN_SEC=average_TIME_per_dataset/1000
sd_time_per_dataset=(np.std(np.array([TIME1, TIME2, TIME3,TIME4,TIME5])))/1000

# #Export measures 
measures={"Name:": name, "Sig_cutoff": 0.70, "Traintime per dataset": TIME_IN_SEC, "SD Time per dataset=": sd_time_per_dataset, "kernel width": width_of_kernel, "R2 Y": average_r2,"R2_SD":all_r2_sd,"R2_arrays_avgs":all_r2_averages, "MATRIX1 ": all_matrices[0], "m1 per pred": all_m1[0], "MATRIX2: ": all_matrices[1], "m2 per pred": all_m1[1],"MATRIX3: ": all_matrices[2], "m3 per pred": all_m1[2],"MATRIX4: ": all_matrices[3],"m4 per pred": all_m1[3],"MATRIX5: ": all_matrices[4], "m5 per pred": all_m1[4],"Sign_per_Dataset pred1: ": all_significanceweights1, "Sign_per_Dataset pred2: ": all_significanceweights2}
# #measures={"Name:": name, "Sig_cutoff": 0.70, "R2 Y": average_r2,"R2_SD":all_r2_sd,"R2_arrays_avgs":all_r2_averages, "Sign_per_Dataset pred1: ": all_significanceweights1, "Sign_per_Dataset pred2: ": all_significanceweights2, "MATRIX1 ": all_matrices[0],  "m1 per pred": all_m1[0],"MATRIX2 ": all_matrices[1], "m2 per pred": all_m1[1]}
with open('C:/Users/irisv/Desktop/path/end/filename_for_export_multivariate.csv', 'w') as csv_file: 
    writer = csv.writer(csv_file)
    for key, value in measures.items():
        writer.writerow([key, value])

# For five datasets the sequence of the significance weights is the following,
# Dataset1: all_significanceweights1[0], all_significanceweights1[1], all_significanceweights1[2],
# Dataset2: all_significanceweights1[3], all_significanceweights1[4], all_significanceweights1[5], and so on
# all_significanceweights1[0] contains three scores with the first score representing the significance of predictor 0 (as defined with the inner loop j)
# for target 0, the second represents the same but for target 1 and the third one represents the significance of predictor for target 2