# kul-EEG-sourceconnectivity
Code for Master's thesis in Artificial Intelligence: Using Neural Networks to derive Directed Connectivity between Reconstructed EEG-Sources.


Python Scripts for univariate (runCNND_Basic, CNN2d_Basic) versus multivariate connectivity predictions (runCNNd_multivar, CNNd_multivar). With regard to multivariate predictions, the current code can currently be run with two or three predictors (for 3 predictors: by uncommenting 'X2' and reshaping X2 where necessary, following how X1 is reshaped to fit the supported format) and this can be extended. In the code itself, information regarding the amount of predictors is provided.

Analysis: In runCNND_Basic, calculations of True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN), Precision, Recall and F1-score are provided, given that the Ground Truth is entered as well in runCNND_Basic. One can choose a full versus partial Ground Truth analysis: in the partial analysis
self-connectivity of a brain source is not included in the final calculations.

Data: NeuralNets_sample_data.zip contains 10 simulated EEG-datasets to be separated in two conditions:
1) 5 data sets in which the active brain sources are located superficially in the brain
2) 5 data sets in which the brain sources are located deep in the brain.
The first row of data in the files is neglected, unless specified otherwise.

These data sets were created using an adaptation of the simulation Framework (implemented in Matlab) of Anzolin et al (2019):
Quantifying the Effect of Demixing Approaches on Directed Connectivity Estimated Between Reconstructed EEG Sources.

Citation: Faes, A., Vantieghem, I., Van Hulle, M.M. (2022). Neural Networks for Directed Connectivity
Estimation in Source-Reconstructed EEG Data. Applied Sciences, 12(6), 2889.
