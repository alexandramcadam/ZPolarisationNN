"""
Alexandra McAdam 
16/11/24

Analyse MI for DNN feature importance 
"""

import pandas as pd 
import h5py
import numpy as np
from process import X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, name, cols_dnn, inv_mass_test
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

#calculate and normalise MI 

#test data 
#mi = mutual_info_classif(X_test_tensor, y_test_tensor, random_state=42)

#train data 
#mi = mutual_info_classif(X_train_tensor, y_train_tensor, random_state=42)

#MI with DNN output 
name = '6v2'
path = f'../modelsDNN/modeloutputsDNN/result_{name}.h5' 
f = h5py.File(path, 'r')
y_preds =  pd.DataFrame(f['preds']) 


X = X_test_tensor[:, cols_dnn]
y = np.array(y_preds[0]).ravel()

#mi = mutual_info_classif(X, y, random_state=42)
mi = mutual_info_classif(X, y_test_tensor, random_state=42)

mi_normalized = MinMaxScaler().fit_transform(mi.reshape(-1, 1))

#print results 
feature_names = np.array([
    "FJ_eta",
    "FJ_flavour",
    "FJ_mass",
    "FJ_pT",
    "FJ_phi",
    "LeadingSubJet_Eta",
    "LeadingSubJet_Phi",
    "LeadingSubJet_pT",
    "Lep_pT_balance",
    "NegLep_Eta",
    "NegLep_Phi",
    "NegLep_pT",
    "Phi",
    "Phi1",
    "PosLep_Eta",
    "PosLep_Phi",
    "PosLep_pT",
    "SubLeadingSubJet_Eta",
    "SubLeadingSubJet_Phi",
    "SubLeadingSubJet_pT",
    "Vlep_eta",
    "Vlep_mass",
    "Vlep_pT",
    "Vlep_phi",
    "cosThetaStar",
    "costheta1",
    "costheta2"
])

new_feature_names = feature_names[cols_dnn]

feature_importances = pd.DataFrame({'Feature': new_feature_names, 'Importance': mi_normalized.flatten()})
#feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': mi.flatten()})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)
        



