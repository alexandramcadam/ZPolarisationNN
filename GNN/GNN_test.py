"""
Alexandra McAdam 
11/10/24

Testing GNN Model
"""


import torch
import torch.optim as optim
from torch import nn
from process import weights
from module import ParticleNet
from neural import NeuralNetGraph
from scoring import accept_epoch, reject_epoch
from evaluation import model_eval
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reje_e = EpochScoring(reject_epoch, lower_is_better=False)
    acce_e = EpochScoring(accept_epoch, lower_is_better=False)
    stop = EarlyStopping(monitor='auc_epoch', patience=3, lower_is_better=False, threshold=-0.001, load_best=True)
    cp = Checkpoint(monitor='auc_epoch_best')

    #best parameters from course seach lr and dropout 
    kernel_sizes = [64, 128, 256, 512]
    fc_size = 128 #128 original, ParticleNet uses 256 
    dropout = 0.3 
    k = 16 #number of nearest neighbours in graph 
    
    #Example of training model for certain configuration without DisCo
    model_17 = NeuralNetGraph(
        module = ParticleNet(kernel_sizes, fc_size, dropout, k, node_feat_size=9, num_classes=len(weights)), 
        criterion = nn.CrossEntropyLoss,
        optimizer = optim.Adam,
        criterion__reduction = 'none',
        criterion__weight = weights,
        verbose = 10,
        optimizer__lr = 0.005, #learning rate 
        batch_size= 256,
        classes = [0,1],
        train_split=None,
        device = device,
        max_epochs = 10, 
        iterator_train__num_workers = 6, 
        iterator_valid__num_workers = 6,
        iterator_train__pin_memory = False,
        iterator_valid__pin_memory = False,
        callbacks = [reje_e, acce_e, stop, cp] 
    )

    model_17.initialize()
    model_17.load_params(f_params='../models/model_17.pkl') #change to match where model stored 


    models = [model_17]
    model_name = ['17']
    
    #discriminant and score is evaluated on test dataset
    model_eval(models, model_name) 







