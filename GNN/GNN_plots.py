"""
Alexandra McAdam 
11/10/24

Plotting GNN model testing results - discriminant, ROC, confusion matrix 
"""

import h5py
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns

#from process import weights, test_dataset, test_inv_mass
#from process_labframe import weights, test_dataset, test_inv_mass, SliceDataset
from process_restframeVV import weights, test_dataset, test_inv_mass, SliceDataset


if __name__ == '__main__':

    model_num = '14' #change for which model you want - remember to match with dataset 

    #plotting model output 
    path = f'../models/modeloutputs/result_{model_num}.h5'
    f = h5py.File(path, 'r')

    #test results 
    preds =  pd.DataFrame(f['preds']) #predictions 
    score =  pd.DataFrame(f['score']) #probabilities 

    f_values = 1/weights
    multiply = np.array(f_values)*np.array(score) + 1e-8
    frac = multiply[:,0]/multiply[:,1]
    D_val = np.log(frac) #discriminant 


    #data truth labels 
    labels = [] 
    for i in range(len(test_dataset)):
        labels.append(test_dataset[i].y.item())

    accuracy = accuracy_score(labels, preds, normalize = True)
    b_accuracy = balanced_accuracy_score(labels, preds)
    print('accuracies')
    print(accuracy, b_accuracy)


    #colour scheme 
    layers = [1, 1, 1, 1, 1] 

    # Get the colormap
    cmap = plt.get_cmap('viridis')

    # Generate node colors for each layer
    colors = []
    for i, layer_size in enumerate(layers):
        colors.extend([cmap(i / (len(layers) - 1))] * layer_size)
    mainc = colors[1]
    secondc = colors[3]
    thirdc = colors[2]


    #plotting discriminant - log ratio 
    log_ratio = np.log(score[0] / score[1]) 

    #plotting log ratio with 2 distributions overlayed 
    labels = np.array(labels)
    mask0 = labels == 0
    mask1 = labels == 1

    dis0 = log_ratio[mask0]
    dis1 = log_ratio[mask1]

    plt.figure(figsize=(8,6))
    plt.hist(dis0, density=True, histtype='step', bins=100, color=mainc, alpha=0.7, label='$\mathrm{Z}_L\mathrm{Z}_L$')
    plt.hist(dis1, density=True, histtype='step', bins=100, color=secondc, alpha=0.7, label='$\mathrm{Z}_T\mathrm{Z}_T$')
    plt.xlabel('D$_{\mathrm{Z}_L\mathrm{Z}_L}$', loc='right')
    plt.ylabel('Event Fraction', loc='top')
    #plt.ylim(0.0,1.0)
    #plt.title(f'Log Ratio of ZlZl vs ZtZt Probabilities (Model {model_num})', pad=30)
    plt.legend()
    #plt.grid(True)
    #plt.savefig(f'../models/finalplots/discriminant_m{model_num}.png', dpi=600)
    #plt.show()
    
    
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
    #plt.savefig(f'../models/finalplots/roc_m{model_num}.png', dpi=600)
    #plt.show()


    #plotting confusion matrix 
    cm = confusion_matrix(labels, preds, normalize= 'true')
    legend = ['$\mathrm{Z}_L\mathrm{Z}_L$','$\mathrm{Z}_T\mathrm{Z}_T$'] #check correct way around 

    plt.figure(figsize=(8,6))
    sns.heatmap(cm*100, annot=True, fmt='.0f', xticklabels=legend, yticklabels=legend, cmap='Greens', vmin=0, vmax=100, cbar_kws={'label': 'True and False Rates [%]'}) #blues is standard
    plt.xlabel('Predicted Label', loc='right')
    plt.ylabel('Truth Label', loc='top')
    #plt.title(f'Confusion Matrix (Model {model_num})', pad=30)
    #plt.savefig(f'../models/finalplots/cm_m{model_num}.png', dpi=600)
    #plt.show()


