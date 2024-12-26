"""
Alexandra McAdam 
23/11/24

Find GNN best operating point and plot for different invariant masses
"""


import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt

#from process_labframe import weights, test_dataset, test_inv_mass
from process_restframeVV import weights, test_dataset, test_inv_mass, inv_mass, dataset


model_num = '14' #change for which model you want - remember to match with dataset 

#plotting model output 
path = f'../models/modeloutputs/result_{model_num}.h5'
f = h5py.File(path, 'r')

#data
#d_val = pd.DataFrame(f['D_val']) #not actually used - code not altered 
preds =  pd.DataFrame(f['preds'])
score =  pd.DataFrame(f['score'])


#data truth labels 
labels = [] 
for i in range(len(test_dataset)):
    labels.append(test_dataset[i].y.item())


#colour scheme 
layers = [1, 1, 1, 1, 1]  # Input layer, 3 hidden layers, output layer

# Get the colormap
cmap = plt.get_cmap('viridis')

# Generate node colors for each layer
colors = []
for i, layer_size in enumerate(layers):
    colors.extend([cmap(i / (len(layers) - 1))] * layer_size)
mainc = colors[1]
secondc = colors[3]
thirdc = colors[2]

#plt.hist(test_inv_mass, bins=100)
#plt.title('masses')


#plotting discriminant - log ratio 
log_ratio = np.log(score[0] / score[1]) #no weights 
#log_ratio = D_val #weighted 

#plotting log ratio with 2 distributions overlayed 
labels = np.array(labels)
mask0 = labels == 0
mask1 = labels == 1

dis0 = log_ratio[mask0]
dis1 = log_ratio[mask1]




#combine data and labels
discriminants = np.concatenate([dis0, dis1])
labels = np.concatenate([np.ones_like(dis0), np.zeros_like(dis1)])

#sort discriminant values
thresholds = np.linspace(min(discriminants), max(discriminants), 100)

balanced_accuracies = []
for T in thresholds:
    #classify based on threshold
    predictions = (discriminants > T).astype(int)  #predict class 0 if discriminant > T
    
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
desired_balanced_accuracy = 1 #just want best operating point 
best_threshold_index = np.argmin(np.abs(np.array(balanced_accuracies) - desired_balanced_accuracy))
best_threshold = thresholds[best_threshold_index]

print(f"Best threshold for {desired_balanced_accuracy*100}% balanced accuracy: {best_threshold}")

#plot distributions of discriminants
plt.figure(figsize=(8, 6))

bins = np.linspace(min(discriminants), max(discriminants), 50)

plt.hist(dis0, density=True, histtype='step', bins=100, color=mainc, label='$\mathrm{Z}_L\mathrm{Z}_L$')
plt.hist(dis1, density=True, histtype='step', bins=100, color=secondc,  label='$\mathrm{Z}_T\mathrm{Z}_T$')
plt.xlabel('D$_{\mathrm{Z}_L\mathrm{Z}_L}$', loc='right')
plt.ylabel('Event Fraction', loc='top')

#highlight the best threshold
plt.axvline(best_threshold, color='indianred', linestyle='--', label=f'Best Operating Point')

mean_best_threshold = best_threshold
# Add labels and legend
plt.legend(fontsize=10)
plt.xlim(-5,5)
#plt.title('Discriminant Distributions with Threshold', fontsize=14)
#plt.grid(alpha=0.3)
#plt.tight_layout()
#plt.savefig(f'../models/finalplots/discriminant_m{model_num}.png', dpi=600)


#now finding best operating point for mass bins

invariant_masses = test_inv_mass
discriminants = np.concatenate([dis0, dis1])
labels = np.concatenate([np.ones_like(dis0), np.zeros_like(dis1)])

#cut-off values for invariant mass - need reasonable statistics 
min_mass = 180  
max_mass = 2000  

#define invariant mass bins within the cut-off range
mask_mass = (invariant_masses >= min_mass) & (invariant_masses <= max_mass)
mass_bins = np.linspace(min_mass, max_mass, 50)  
bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2  

#desired balanced accuracies
desired_balanced_accuracies = [1]  
thresholds_dict = {b: [] for b in desired_balanced_accuracies}  

#iterate over invariant mass bins
for i in range(len(mass_bins) - 1):
    #filter events within the current mass bin and cut-off range
    bin_mask = (invariant_masses >= mass_bins[i]) & (invariant_masses < mass_bins[i + 1])
    discriminants_bin = discriminants[bin_mask]
    labels_bin = labels[bin_mask]
    
    if len(discriminants_bin) == 0:  #skip if empty 
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

#plot thresholds vs. invariant mass bins for multiple balanced accuracies
plt.figure(figsize=(8, 6))

for b_acc, thresholds in thresholds_dict.items():
    #plt.step(bin_centers, thresholds, linestyle='-', label=f'Balanced Accuracy {b_acc}', color='indianred')
    plt.step(bin_centers, thresholds, linestyle='-', color='lightpink')
    plt.axhline(mean_best_threshold, linestyle='--', color='indianred', label='Best Operating Point')

# Add labels, legend, and grid
plt.xlabel('Invariant Mass $m_{ZZ}$ [GeV]', loc='right')
plt.ylabel('Operating Point', loc='top')
#plt.title('Thresholds for Desired Balanced Accuracies', fontsize=14)
#plt.grid(alpha=0.3)
plt.legend(fontsize=10)
#plt.savefig(f'../models/finalplots/op_m{model_num}.png', dpi=600)


#plot input distribution of masses 
all_labels = [] 
for i in range(len(dataset)):
    all_labels.append(dataset[i].y.item())

all_labels = np.array(all_labels)

plt.figure(figsize=(8, 6))
plt.hist(inv_mass[all_labels==0], color=mainc, histtype='step', bins=mass_bins, label='$\mathrm{Z}_L\mathrm{Z}_L$')
plt.hist(inv_mass[all_labels==1], color=secondc, histtype='step', bins=mass_bins, label='$\mathrm{Z}_T\mathrm{Z}_T$')

plt.xlabel('Invariant Mass $m_{ZZ}$ [GeV]', loc='right')
plt.ylabel('Number of Events', loc='top')

plt.legend()
plt.savefig(f'../models/finalplots/inputs_m{model_num}.png', dpi=600)

plt.show()

