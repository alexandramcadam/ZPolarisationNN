"""
Alexandra McAdam 
16/11/24

Process Monte Carlo sim data in .h5 file for DNN 
"""


import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


pd.options.mode.chained_assignment = None  #default='warn'

hfivesdir = "../datasets/new_Input_Polarisations_ZZ__16th_November2024.h5" #absolute path to dataset 

#open the h5 file containing jet kinematic information
with h5py.File(hfivesdir) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])
    
#relabel polarisations 
df.replace({'polarisation_type':{1:0, 2:1}}, inplace=True) 

#relabel flavour 
df.replace({'flavour':{0:0, 6:1, 10:2, 23:3, 24:3, 42:4}}, inplace=True) 
df.dropna(inplace=True)

#set the polarisation_type label as the truth record for training
X = df.drop(['polarisation_type'], axis=1) #all columns for initial training -> then need to use MI to drop unimportant features 
y = df['polarisation_type']

#adding oversampling to balance datasets 
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y) 

#print(y.value_counts())
#print(y_resampled.value_counts())


#split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=1234)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#normalise the dataset 
cols_to_normalize = X_train.select_dtypes(include='number').columns.to_list()  
scaler = MinMaxScaler()
scaler.fit(X_train[cols_to_normalize])

X_train[cols_to_normalize] = scaler.transform(X_train[cols_to_normalize])
X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#columns that will be needed for inverting the normalisation
cols_inv = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] #all cols 

#columns that will be used as the network input
#cols_dnn = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] #all
#cols_dnn = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26] 
#cols_dnn = [0,1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,24,25,26] 
#cols_dnn = [0,1,2,3,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,24,25,26] 
cols_dnn = [0,1,2,3,5,6,7,9,11,12,14,15,16,17,18,19,20,21,22,24,25,26] #best validation accuracy is this (from MI analysis) 
#cols_dnn = [0,1,2,3,5,6,7,9,11,12,14,15,16,17,18,19,20,22,24,25,26] 

name = "6oversample" #name of the model 
weights = torch.tensor([1.0,1.0]) #since resampled 

#invariant mass 
def invariant_mass(FJ_pT, FJ_eta, FJ_phi, Vlep_pT, Vlep_eta, Vlep_phi):
    # Four-momentum components for FJ boson
    FJ_px = FJ_pT * np.cos(FJ_phi)
    FJ_py = FJ_pT * np.sin(FJ_phi)
    FJ_pz = FJ_pT * np.sinh(FJ_eta)
    FJ_E = df['FJ_pT']*np.cosh(df['FJ_eta']) #for high energies 

    # Four-momentum components for Vlep boson
    Vlep_px = Vlep_pT * np.cos(Vlep_phi)
    Vlep_py = Vlep_pT * np.sin(Vlep_phi)
    Vlep_pz = Vlep_pT * np.sinh(Vlep_eta)
    Vlep_E = df['Vlep_pT']*np.cosh(df['Vlep_eta'])

    # Total four-momentum
    total_E = FJ_E + Vlep_E
    total_px = FJ_px + Vlep_px
    total_py = FJ_py + Vlep_py
    total_pz = FJ_pz + Vlep_pz
    
    # Invariant mass
    invariant_mass = np.sqrt(total_E**2 - (total_px**2 + total_py**2 + total_pz**2))
    return invariant_mass

inv_mass = invariant_mass(X_resampled['FJ_pT'],X_resampled['FJ_eta'],X_resampled['FJ_phi'], X_resampled['Vlep_pT'],X_resampled['Vlep_eta'],X_resampled['Vlep_phi'])
inv_mass_train, inv_mass_test = train_test_split(inv_mass, test_size=0.1, random_state=1234) 


