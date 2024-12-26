"""
Alexandra McAdam 
11/10/24

Creating file with GNN results 
"""

import torch
import h5py

#match process to dataset used to train model 
from process import weights, test_dataset
#from process_labframe import weights, test_dataset
#from process_restframeVV import weights, test_dataset


#function for producing h5 files containing discriminant and GNN score
#the files can then be processed into results and graphs.
def model_eval(models, model_name):
    X_test_ = test_dataset
    
    for (model, name) in zip(models, model_name):
        with torch.no_grad():
            y_score = model.predict_proba(X_test_)
            y_preds = model.predict(X_test_)
            
        f_values = 1/weights
        multiply = f_values*y_score + 1e-8
        summed = sum(multiply[:,i] for i in range(len(weights)))
        D_val = torch.log(multiply[:,-1]/(summed - multiply[:,-1]))
        D_val = D_val.cpu().numpy()
            
        hf = h5py.File(f'../models/modeloutputs/result_{name}.h5', 'w') #change path for where results stored  
        hf.create_dataset('D_val', data=D_val)
        hf.create_dataset('preds', data=y_preds)
        hf.create_dataset('score', data=y_score)
        hf.close()

