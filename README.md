# ZPolarisationNN

Repository for my senior honours project "Distinguishing Longitudinally and Transversely Polarised Weak Bosons using Machine Learning". A Graph Neureal Network (GNN) and Deep Neural Network (DNN) were used to distinguish the polarisations of Z bosons. Training was performed using a simulated, but experimentally-realistic, dataset of $pp \rightarrow ZZ \rightarrow q\bar{q}\ell\bar{\ell}$ events from $\sqrt{s}=14$ TeV pp collisions. The code is largely adapted from the repository https://github.com/Satriawidy/HggNN. The report for this project is included in the repository for futher information.  

# Set Up

This project was implemented within a virtual environment using python 3 (specifically version 3.9.6) and the dependencies given in `requirements.txt`. 

# GNN
1. Process the input dataset into a suitable format for GNN using one of `process.py`,`process_labframe.py` or `process_restframeVV.py` depending on the input dataset and desired reference frame to train the model. This creates a file with the graph data. 
2. Train the GNN using `GNN_train.py`.
3. Evaluate the GNN performance using `GNN_test.py` which creates a file with model results. 
4. Plot the GNN performance metrics using `GNN_plots.py` and find the best operating point using `GNN_operatingpoints.py`.

# DNN
1. Process the input dataset into a suitable format for the DNN using `process.py` which also performs oversampling of the dataset to get balanced classes. 
2. Train the DNN using `DNN_ZZ.py`.
3. Evaluate the DNN performance using `DNN_test.py` which creates a file with model results. 
4. Plot the DNN performance metrics and find the best operating point using `DNN_plots.py`.
5. Feature importance was analysed using Mutual Information in `DNN_MI.py` to select the optimal input features.
