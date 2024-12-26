"""
Alexandra McAdam 
16/11/24

Plotting DNN test results
"""

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix
from process import X_test_tensor, y_test_tensor, inv_mass_test, name, inv_mass, y_resampled, y, X_resampled, cols_dnn


#plotting model output 
print(name)
path = f'../modelsDNN/modeloutputsDNN/result_{name}.h5'
f = h5py.File(path, 'r')

#data
preds =  pd.DataFrame(f['preds'])
score =  pd.DataFrame(f['score'])

#data truth labels - y_test_tensor
labels = np.array(y_test_tensor) 
# print(len(labels))
# print(len(X_test_tensor))

#colour scheme 
layers = [1, 1, 1, 1, 1] 
cmap = plt.get_cmap('viridis')
colors = []
for i, layer_size in enumerate(layers):
    colors.extend([cmap(i / (len(layers) - 1))] * layer_size)
mainc = colors[1]
secondc = colors[3]
thirdc = colors[2]



#ROC
#plotting ROC curve 
y_preds = score[1] #probabilities for ZtZt - roc_curve function uses 1 class probabilities 

fpr, tpr, thresholds = roc_curve(labels, y_preds) 
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color=mainc)
plt.plot(np.linspace(0,1), np.linspace(0,1), linestyle='--', color='lightgrey' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate', loc='top')
plt.xlabel('False Positive Rate', loc='right')
#plt.title(f'ROC Curve (Model {model_num})', pad=30)
plt.legend()
#plt.savefig(f'../modelsDNN/finalplots/roc_m{name}.png', dpi=600)
#plt.show()




#Discriminant 
#plotting discriminant - log ratio 
log_ratio = np.log(score[0] / score[1]) #no weights 
#log_ratio = D_val #weighted 

#plotting log ratio with 2 distributions overlayed 
labels = np.array(labels)
mask0 = labels == 0
mask1 = labels == 1

dis0 = log_ratio[mask0]
dis1 = log_ratio[mask1]




#Best operating point and variation with masses 

#combine data and labels
discriminants = np.concatenate([dis0, dis1])
labels = np.concatenate([np.ones_like(dis0), np.zeros_like(dis1)])

thresholds = np.linspace(min(discriminants), max(discriminants), 100) #threshold is the operating point 

balanced_accuracies = []
for T in thresholds:
    #classify based on threshold
    predictions = (discriminants > T).astype(int)  
    
    #calculate TP, TN, FP, FN
    TP = np.sum((predictions == 0) & (labels == 0))
    TN = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 0) & (labels == 1))
    FN = np.sum((predictions == 1) & (labels == 0))
    
    #compute TPR and TNR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    #compute balanced accuracy
    balanced_accuracy = (TPR + TNR) / 2
    balanced_accuracies.append(balanced_accuracy)

#find threshold for desired balanced accuracy
desired_balanced_accuracy = 1
best_threshold_index = np.argmin(np.abs(np.array(balanced_accuracies) - desired_balanced_accuracy))
best_threshold = thresholds[best_threshold_index]

print(f"Best threshold for {desired_balanced_accuracy*100}% balanced accuracy: {best_threshold}")

#plot distributions of discriminants
plt.figure(figsize=(8, 6))

bins = np.linspace(min(discriminants), max(discriminants), 50)

plt.hist(dis0, density=True, histtype='step', bins=100, color=mainc, label='$\mathrm{Z}_L\mathrm{Z}_L$') #alpha=0.7
plt.hist(dis1, density=True, histtype='step', bins=100, color=secondc, label='$\mathrm{Z}_T\mathrm{Z}_T$')
plt.xlabel('D$_{\mathrm{Z}_L\mathrm{Z}_L}$', loc='right')
plt.ylabel('Event Fraction', loc='top')

#highlight the best threshold
plt.axvline(best_threshold, color='indianred', linestyle='--', label=f'Best Operating Point')

mean_best_threshold = best_threshold
plt.legend(fontsize=10)
plt.xlim(-7,7)
#plt.title('Discriminant Distributions with Threshold', fontsize=14)
#plt.grid(alpha=0.3)
#plt.tight_layout()
plt.savefig(f'../modelsDNN/finalplots/discriminant_m{name}.png', dpi=600)


#now plot best operating point for given mass bin 

invariant_masses = inv_mass_test
discriminants = np.concatenate([dis0, dis1])
labels = np.concatenate([np.ones_like(dis0), np.zeros_like(dis1)])

#cut-off values for invariant mass
min_mass = 180 
max_mass = 2000  

#define invariant mass bins within the cut-off range
mask_mass = (invariant_masses >= min_mass) & (invariant_masses <= max_mass)
mass_bins = np.linspace(min_mass, max_mass, 50)  
bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2  

#list of desired balanced accuracies
desired_balanced_accuracies = [1]  
thresholds_dict = {b: [] for b in desired_balanced_accuracies}  

#iterate over invariant mass bins
for i in range(len(mass_bins) - 1):
    #filter events within the current mass bin and cut-off range
    bin_mask = (invariant_masses >= mass_bins[i]) & (invariant_masses < mass_bins[i + 1])
    discriminants_bin = discriminants[bin_mask]
    labels_bin = labels[bin_mask]
    
    if len(discriminants_bin) == 0:  #skip empty bins
        for b in desired_balanced_accuracies:
            thresholds_dict[b].append(np.nan)
        continue
    
    #compute thresholds for each desired balanced accuracy
    thresholds_bin = np.linspace(min(discriminants_bin), max(discriminants_bin), 50)
    for b_acc in desired_balanced_accuracies:
        balanced_accuracies = []
        for T in thresholds_bin:
            predictions = (discriminants_bin > T).astype(int)  
            
            #calculate TP, TN, FP, FN
            TP = np.sum((predictions == 1) & (labels_bin == 1))
            TN = np.sum((predictions == 0) & (labels_bin == 0))
            FP = np.sum((predictions == 1) & (labels_bin == 0))
            FN = np.sum((predictions == 0) & (labels_bin == 1))
            
            #compute TPR and TNR
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
            
            #compute balanced accuracy
            balanced_accuracy = (TPR + TNR) / 2
            balanced_accuracies.append(balanced_accuracy)
        
        #find threshold for the current balanced accuracy
        best_threshold_index = np.argmin(np.abs(np.array(balanced_accuracies) - b_acc))
        best_threshold = thresholds_bin[best_threshold_index]
        thresholds_dict[b_acc].append(best_threshold)

plt.figure(figsize=(8, 6))

for b_acc, thresholds in thresholds_dict.items():
    #plt.step(bin_centers, thresholds, linestyle='-', label=f'Balanced Accuracy {b_acc}', color='indianred')
    plt.step(bin_centers, thresholds, linestyle='-', color='lightpink')
    plt.axhline(mean_best_threshold, linestyle='--', color='indianred', label='Best Operating Point')


plt.xlabel('Invariant Mass $m_{ZZ}$ [GeV]', loc='right')
plt.ylabel('Operating Point', loc='top')
#plt.title('Thresholds for Desired Balanced Accuracies', fontsize=14)
#plt.grid(alpha=0.3)
plt.legend(fontsize=10)
#
#plt.savefig(f'../modelsDNN/finalplots/op_m{name}.png', dpi=600)


plt.figure(figsize=(8, 6))
plt.hist(inv_mass_test[mask_mass], color=thirdc, histtype='step', bins=mass_bins)
plt.xlabel('Invariant Mass $m_{ZZ}$ [GeV]', loc='right')
plt.ylabel('Number of Events', loc='top')
#plt.savefig(f'../models/finalplots/masses_m{name}.png', dpi=600)





#Input distributions for each DNN feature 

all_labels = np.array(y_resampled)

#mass
plt.figure(figsize=(8, 6))
plt.hist(inv_mass[all_labels==0], color=mainc, histtype='step', bins=mass_bins, label='$\mathrm{Z}_L\mathrm{Z}_L$')
plt.hist(inv_mass[all_labels==1], color=secondc, histtype='step', bins=mass_bins, label='$\mathrm{Z}_T\mathrm{Z}_T$')

plt.xlabel('Invariant Mass $m_{ZZ}$ [GeV]', loc='right')
plt.ylabel('Number of Events', loc='top')

plt.legend()
#plt.savefig(f'../modelsDNN/finalplots/inputs_m{name}.png', dpi=600)


#for each feature 
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

all_labels = np.array(y_test_tensor)
dis0 = all_labels==0
dis1 = all_labels==1


X = X_test_tensor[:, cols_dnn]

for i, v_name in enumerate(new_feature_names):
    plt.figure(figsize=(8, 6))

    plt.hist(np.array(X[dis0])[:,i], density=True, color=mainc, histtype='step', bins=100, label='$\mathrm{Z}_L\mathrm{Z}_L$')
    plt.hist(np.array(X[dis1])[:,i], density=True, color=secondc, histtype='step', bins=100,  label='$\mathrm{Z}_T\mathrm{Z}_T$')

    plt.xlabel(f'{v_name}', loc='right')
    plt.ylabel('Event Fraction', loc='top')

    plt.legend()
    plt.savefig(f'../modelsDNN/finalplots/input_{v_name}_m{name}.png', dpi=600)

plt.show()
