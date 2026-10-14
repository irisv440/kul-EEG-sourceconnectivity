# \# kul-EEG-sourceconnectivity

# 

# Code for my Master's thesis in Artificial Intelligence (KU Leuven, 2021): \*Using Neural Networks to derive Directed Connectivity between Reconstructed EEG Sources\*.

# 

# The results were published in:

# Faes, A., Vantieghem, I., \& Van Hulle, M. M. (2022). Neural Networks for Directed Connectivity Estimation in Source-Reconstructed EEG Data. \*Applied Sciences\*, 12(6), 2889. https://doi.org/10.3390/app12062889

# 

# This repository contains the code as used for the thesis (2021–2022) and is not actively maintained. In 2026, file paths and data loading were updated so that the scripts run without changes.

# 

# \## Contents

# 

# \- `runCNND\_Basic.py` and `CNN2d\_Basic.py`: univariate connectivity predictions.

# \- `runCNNd\_multivar.py` and `CNNd\_multivar.py`: multivariate connectivity predictions. The code can be run with two or three predictors (for three predictors: uncomment `X2` and reshape it where necessary, following how `X1` is reshaped). This can be extended; the code itself documents the number of predictors.

# \- `NeuralNets\_sample\_data.zip`: 10 simulated EEG datasets in two conditions:

# &#x20; 1. 5 datasets in which the active brain sources are located superficially (`FarSuperficial`)

# &#x20; 2. 5 datasets in which the brain sources are located deep in the brain (`FarDeep`)

# 

# &#x20; The first row of data in the files is neglected, unless specified otherwise. The datasets were created with an adaptation of the simulation framework (implemented in MATLAB) of Anzolin et al. (2019): \*Quantifying the Effect of Demixing Approaches on Directed Connectivity Estimated Between Reconstructed EEG Sources\*.

# 

# \## Requirements

# 

# Tested with Python 3.7 and TensorFlow 2.9. Using a separate environment is recommended.

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \## How to run

# 

# 1\. Unzip `NeuralNets\_sample\_data.zip` in the repository folder, so that the data are located in `NeuralNets\_sample\_data/FarSuperficial/` and `NeuralNets\_sample\_data/FarDeep/`. On Windows, "Extract All" may create an extra nested folder; make sure the path matches the one above.

# 2\. Run the univariate analysis (uses the superficial datasets by default):

# ```bash

# &#x20;  python runCNND\_Basic.py

# ```

# 3\. Run the multivariate analysis (uses the deep datasets by default):

# ```bash

# &#x20;  python runCNNd\_multivar.py

# ```

# 

# To switch between the superficial and deep datasets, change `data\_dir` and the dataset block at the top of each script.

# 

# The scripts train a network for every pair of sources, over 5 runs per dataset and 5 datasets, so a full run can take a while on a CPU.

# 

# \## Analysis and output

# 

# `runCNND\_Basic.py` calculates true positives (TP), false positives (FP), true negatives (TN), false negatives (FN), precision, recall and F1-score by comparing the results with the ground truth defined in the script. Both a full analysis and a partial analysis are provided; in the partial analysis, self-connectivity of a brain source is not included in the final calculations. Results are printed to the console. To also export them to a CSV file in `output/`, uncomment one of the two `measures` blocks (five datasets or two test datasets) and the `with open(...)` block at the end of the script.

# 

# `runCNNd\_multivar.py` writes its results to `output/<name>.csv`; set `name` at the top of the script.

