"""
Alexandra McAdam 
16/11/24

Training DNN  
"""

import torch
import torch.optim as optim
from torch import nn
from process import weights, X_train_tensor, y_train_tensor, cols_dnn, name
from module import DNN, DNNMini
from neural import NeuralNetBase
from scoring import accept_epoch, reject_epoch
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Constructing callbacks for monitoring the training 
    reje_e = EpochScoring(reject_epoch, lower_is_better=False)
    acce_e = EpochScoring(accept_epoch, lower_is_better=False)
    stop = EarlyStopping(monitor='valid_acc', patience=5, lower_is_better=False, threshold=0.0001, load_best=True)
    cp = Checkpoint(monitor='valid_acc_best')


    #using similar parameters as GNN  
    model = NeuralNetBase(
                        module = DNNMini([64,64,64,64], 0.3, len(cols_dnn), len(weights)), #kernels, dropout, feature length, number classes 
                        criterion = nn.CrossEntropyLoss,
                        criterion__reduction = 'none',
                        criterion__weight = weights,
                        optimizer = optim.Adam,
                        optimizer__lr = 0.005,
                        batch_size = 256,
                        max_epochs = 30,
                        verbose = 10,
                        device = device,
                        callbacks = [stop, cp]
                        ) 

    
    #training
    model.fit(X_train_tensor, y_train_tensor)
    
    #saving the model for evaluation
    model.save_params(f_params=f'../modelsDNN/model_{name}.pkl') 