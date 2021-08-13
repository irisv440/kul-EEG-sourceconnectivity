# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7  2021
Script to run all possible combinations of one-to-one predictions.
Datasets need to be defined, as well as the ground truth of the data.
Output = score matrices with significant R2-scores, Sensitivity, Precision and F1-scores (plus all related ones such as TP, FP), and runtime (how to export time can be defined below in the file)
Extra outputs: R2 scores, significance weights, standard deviation of R2-scores, standard deviation of runtime if defined for export
Parameters to set: amount of datasets to use (num_datasets), which datasets to use (alldata, in the case others are also defined in the file)
Other parameters that have a default value but can be changed include width_of_kernel (default=4), learning_rate (0.005), significance=0.9998, and
window_size (default =5) i.e. Window that should be considered to preprocess (reshape) the timeseries for a conv2d

@author: irisv
"""

import pandas as pd
import numpy as np
import csv


from CNN2d_Basic import findCONN


def Average(lst):
    return sum(lst) / len(lst)

# Gets full ground truth including self-connectivity. [a,b] with a = predictor and b = target
def get_truth():
    ground_truth = np.zeros((3, 3))
    ground_truth[0, 0] = 1
    ground_truth[0, 1] = 1
    ground_truth[1, 0] = 1
    ground_truth[1, 1] = 1
    ground_truth[2, 1] = 1
    ground_truth[2, 2] = 1
    return ground_truth
# Gets partial ground truth excluding self-connectivity. [a,b] with a = predictor and b = target
def get_partialTruth():
    ground_truth = np.zeros((3, 3))
    ground_truth[0, 1] = 1
    ground_truth[1, 0] = 1
    ground_truth[2, 1] = 1
    print(ground_truth)
    return ground_truth

# Takes in R2-score matrix for one dataset (averaged over the preferred amount of runs)
# Returns TP, FP, TN, FN, Precision, Recall and F1-score,
# after comparison with truth (for int=1 the full truth is used, for int=0 the partial truth is used)
def get_CONNmetrices(matrix, int):   
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if int==1:
        truth=get_truth()
        for row in range(0,3):
            for col in range(0,3):
                if matrix[row][col]!=0 and truth[row][col]==1:
                    TP += 1
                if matrix[row][col]!=0 and truth[row][col]==0: #): # row 2 col 0 means the indirect connection from X3 to X1 and should not be count as FP
                    FP+= 1
                if matrix[row][col]==truth[row][col]==0:
                    TN += 1
                if matrix[row][col]==0 and truth[row][col]==1:
                    FN += 1 
    if int==0:
        truth=get_partialTruth()
        for row in range(0,3):
            for col in range(0,3):
                if matrix[row][col]!=0 and truth[row][col]==1:
                    TP += 1
                if matrix[row][col]!=0 and truth[row][col]==0 and row!=col: #): # row 2 col 0 means the indirect connection from X3 to X1 and should not be count as FP
                    FP+= 1
                if matrix[row][col]==truth[row][col]==0:
                    TN += 1
                if matrix[row][col]==0 and truth[row][col]==1:
                    FN += 1   
    if TP==FP==0:
        precision=0 # division by zero
    else:
        precision = (TP) / (TP + FP)
    if TP==FN==0:
        recall=0 # division by zero
    else:
        recall = (TP) / (TP + FN)
    if precision == recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return TP, FP, TN, FN, recall, precision, f1



# Where should the output be saved and under which name?
name="Filename_with_conditions_params_and_cutoff" 
path= "C:/Users/irisv/Desktop/path/end/"

#Datafiles to use
dataset1=pd.read_csv('DataFarSuperficial1500_1.csv')
dataset2=pd.read_csv('DataFarSuperficial1500_2.csv')
dataset3=pd.read_csv('DataFarSuperficial1500_3.csv')
dataset4=pd.read_csv('DataFarSuperficial1500_4.csv')
dataset5=pd.read_csv('DataFarSuperficial1500_5.csv')


width_of_kernel=4 # time dimension 
learning_rate=0.005
significance=0.9998
window_size=5 # Window that should be considered to preprocess (reshape) the timeseries for a conv2d
seed=1000 # seed to use

# Amount of datasets to use
num_datasets=5
#alldata=[dataset1, dataset2]
alldata=[dataset1, dataset2, dataset3, dataset4, dataset5]
# Amount of runs for each dataset, to average over
runs=5


score_matrix=[]
TPs=[]
FPs=[]
TNs=[]
FNs=[]
recalls=[]
precisions=[]
f1_scores=[]

recallsp=[]
precisionsp=[]
f1_scoresp=[]
TPps=[]
FPps=[]
TNps=[]
FNps=[]

all_matrices=[]
all_r2_sd=[]
all_r2_averages=[]
all_significances=[]
all_times=[]
time_per_dataset=[]
time=0
#validation_loss_over_all_datasets=[]

r2scores=[]
r2_averages=[]
r2_sds=[]
r2_sds_cvs=[]
significanceweights=[]

for num in range(0,num_datasets): 
    data=alldata[num]
    score_matrix = np.zeros((data.shape[0], data.shape[0])) 
    time_per_dataset=0
    
    # Runs_per_dataset to average over
    average=range(0,runs)  
    #How many Timeseries are used? Default = all the ones in file = data.shape[0], but if amount_of_timeseries is specified this can be put in place of data.shape[0]
    #amount_of_timeseries=3
    columns=range(0,data.shape[0])
    
    # Predictors and targets
    amount_of_used_predictors=1
    predictorlist=[0,1,2]
    targetlist=[0,1,2]
       
    for k in range(0,len(targetlist)):
        
        new_target=targetlist[k]
        targetseries=new_target
            
        for j in range(0,len(predictorlist)):
            
            new_predictor=predictorlist[j]
             
            allscores=[]
            testdiff_list=[]
            trainingdiff_list=[]
            significancelist=[]
            totaltime=0
            times_avg5_runs=0
        
            for i in average: # averaging over runs of one dataset
                r2_scorelist, runtime, test_differences, training_differences, testloss, mse_y=findCONN(data, columns, window_size, width_of_kernel, significance, amount_of_used_predictors, new_predictor, predictorlist, targetseries)
                allscores.append(np.array(r2_scorelist)) # if average=5 and folds cv= 6 than allscores=30
                testdiff_list.append(test_differences)
                trainingdiff_list.append(training_differences) 
                totaltime=totaltime+runtime
            times_avg5_runs=times_avg5_runs + totaltime
            time=time+times_avg5_runs
            new_testdiff=np.mean(testdiff_list)
            new_diff=np.mean(trainingdiff_list)
            significance_weight=new_testdiff/new_diff
            if new_testdiff>(new_diff*significance):        
                print('Connectivity over 5 runs is not significant') 
            average_r2 = np.mean(allscores) # take mean of whole list
            r2_sd=np.std(allscores) # sd taken over all r2s (number of r2s= number of folds * number of runs to average over)

            if significance_weight>0.7 or average_r2<0: # 0.70 instead of e.g. 0.4 which will result in less positives
                Connectivity=0
                score_matrix[new_predictor][targetseries]=Connectivity
                print(significance_weight, " (=w not significant) R2_score average =", round(average_r2, 4), " Last Test MSE:", testloss)
            else:
                Connectivity=round(average_r2,4)
                score_matrix[new_predictor][targetseries]=Connectivity
                print("R2_score average =", Connectivity, "Test mse= ", round(testloss,4))
                print("Significance average =", round(significance_weight, 4))
    
            r2_averages.append(average_r2) #
            r2_sds.append(r2_sd) # 
            significanceweights.append(significance_weight)

        all_r2_averages.append(r2_averages)
        all_r2_sd.append(r2_sds)
        all_significances.append(significanceweights)
        
    print(score_matrix)
    TP, FP, TN, FN, Recall, Precision, F1= get_CONNmetrices(score_matrix, 1)
    TPp, FPp, TNp, FNp, Recallp, Precisionp, F1p= get_CONNmetrices(score_matrix, 0)
    
    #Scores according to full truth (including self-connectivity)
    recalls.append(Recall)
    precisions.append(Precision)
    f1_scores.append(F1)
    TPs.append(TP)
    FPs.append(FP)
    TNs.append(TN)
    FNs.append(FN)
    
    # Scores according to partial Truth
    recallsp.append(Recallp)
    precisionsp.append(Precisionp)
    f1_scoresp.append(F1p)
    TPps.append(TPp)
    FPps.append(FPp)
    TNps.append(TNp)
    FNps.append(FNp)
    

    # 1 matrix/dataset
    all_matrices.append(score_matrix)
    all_times.append(time)    

# If only two datasets used (for testing):
# TIME1= all_times[0]
# TIME2= all_times[1]-all_times[0]
# TIME_IN_SEC=((TIME1+TIME2)/len(all_times))/1000
# sd_time_per_dataset=np.std(np.array([TIME1, TIME2]))/1000

# If 5 datasets used:
TIME1= all_times[0]
TIME2= all_times[1]-all_times[0]
TIME3=all_times[2]-TIME2
TIME4=all_times[3]-TIME3
TIME5=all_times[4]-TIME4
average_TIME_per_dataset= (TIME1+TIME2+TIME3+TIME4+TIME5)/5
totaltime_all=np.mean(np.array(all_times))
TIME_IN_SEC=average_TIME_per_dataset/1000
sd_time_per_dataset=(np.std(np.array([TIME1, TIME2, TIME3,TIME4,TIME5])))/1000 # in seconds instead of millisec

## Five datasets: Export 1) measures 2) dataframe actual versus predicted
# measures={"Name:": name,"Traintime per dataset": TIME_IN_SEC, "Cutoff if >":0.70, "SD Time per dataset=": sd_time_per_dataset, "kernel height": height_of_kernel ,"R2_SD averages":all_r2_sd,"R2_averages":all_r2_averages, "M1 ": all_matrices[0], "M2: ": all_matrices[1], "M3: ": all_matrices[2], "M4: ": all_matrices[3],"M5: ": all_matrices[4], "Significance_per_Dataset: ": all_significances,
#             "Recall per set: ": recalls, "Precision per set:": precisions, "F1-scores:": f1_scores, "TP": TPs, "FP": FPs, "TN": TNs,"FN": FNs,"partial Recall per set: ": recallsp, "partial Precision per set:": precisionsp, "partial F1-scores:": f1_scoresp, "partial TP": TPps, "partial FP": FPps, "partial TN": TNps,"partial FN": FNps}
## Testing with 2 datasets:
# measures={"Name:": name,"Traintime per dataset": TIME_IN_SEC, "SD Time per dataset=": sd_time_per_dataset, "kernel height": height_of_kernel,"R2_SD":all_r2_sd,"R2_averages":all_r2_averages, "MATRIX1 ": all_matrices[0], "MATRIX2: ": all_matrices[1], "Sign_per_Dataset: ": all_significances,
#               "Recall per set: ": recalls, "Precision per set:": precisions, "F1-scores:": f1_scores, "TP": TPs, "FP": FPs, "TN": TNs,"FN": FNs,"partial Recall per set: ": recallsp, "partial Precision per set:": precisionsp, "partial F1-scores:": f1_scoresp, "partial TP": TPps, "partial FP": FPps, "partial TN": TNps,"partial FN": FNps}

# with open('C:/Users/irisv/Desktop/path/end/preferred_filename_for_saving.csv', 'w') as csv_file:
#       writer = csv.writer(csv_file)
#       for key, value in measures.items():
#           writer.writerow([key, value])
