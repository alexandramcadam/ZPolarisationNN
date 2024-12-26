"""
Alexandra McAdam 
16/11/24

Testing DNN 
"""

import torch
import h5py
import torch.optim as optim
from torch import nn
from module import DNN, DNNMini
from neural import NeuralNetBase
from process import weights, X_test_tensor, y_test_tensor, cols_dnn, name
from scoring import accept_epoch, reject_epoch
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint


if __name__ == '__main__':


    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reje_e = EpochScoring(reject_epoch, lower_is_better=False)
    acce_e = EpochScoring(accept_epoch, lower_is_better=False)
    stop = EarlyStopping(monitor='valid_acc', patience=3, lower_is_better=False, threshold=-0.001, load_best=True)
    cp = Checkpoint(monitor='valid_acc_best')

    #load model 
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

    
    model.initialize()  # This is important!
    model.load_params(f_params=f'../modelsDNN/model_{name}.pkl')


    #save DNN testing results to file 
    X_test_ = X_test_tensor
    y_test_ = y_test_tensor.numpy()

    with torch.no_grad():
        y_score = model.predict_proba(X_test_)
        y_preds = model.predict(X_test_)

        f_values = 1/weights
        multiply = f_values*y_score + 1e-8
        summed = sum(multiply[:,i] for i in range(len(weights)))
        D_val = torch.log(multiply[:,-1]/(summed - multiply[:,-1]))
        D_val = D_val.cpu().numpy()
        
        hf = h5py.File(f'../modelsDNN/modeloutputsDNN/result_{name}.h5', 'w') 
        hf.create_dataset('D_val', data=D_val)
        hf.create_dataset('preds', data=y_preds)
        hf.create_dataset('score', data=y_score)
        hf.close()
        

