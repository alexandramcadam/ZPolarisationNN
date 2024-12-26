"""
Alexandra McAdam 
11/10/24

GNN models and training 
"""

import torch
import torch.optim as optim
from torch import nn
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint
from module import ParticleNet
from neural import NeuralNetGraph
from scoring import accept_epoch, reject_epoch

#remember to use correct process for dataset and labframe or ZZ frame 
from process import weights, train_dataset
#from process_labframe import weights, train_dataset
#from process_restframeVV import weights, train_dataset


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Constructing callbacks for monitoring the training 
    reje_e = EpochScoring(reject_epoch, lower_is_better=False)
    acce_e = EpochScoring(accept_epoch, lower_is_better=False)
    stop = EarlyStopping(monitor='valid_acc', patience=3, lower_is_better=False, threshold=0.0001, load_best=True) 
    cp = Checkpoint(monitor='valid_acc_best')

    #Dividing the train dataset into train and validation (for monitoring)
    torch.manual_seed(1234)
    train_dataset = train_dataset.shuffle()

    a = int(len(train_dataset))
    b = int(a*0.8)
    X_train = train_dataset[:b]
    X_valid = train_dataset[b:a]

    #best parameters from course seach lr and dropout 
    kernel_sizes = [64, 128, 256, 512]
    fc_size = 128 #128 original, ParticleNet uses 256 
    dropout = 0.3 #dropout probability 
    k = 16 #number of nearest neighbours in graph 
    

    #configure model 
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


    #training
    model_17.fit(X_train, X_valid)

    #saving model for evaluation
    model_17.save_params(f_params='../models/model_17.pkl') #remember change path for new models 
