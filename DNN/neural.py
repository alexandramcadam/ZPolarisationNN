"""
Alexandra McAdam 
16/11/24

Subclass NeuralNetClassifier for DNN  
"""

import torch
import numpy as np
import torch.nn.functional as F
from skorch.dataset import unpack_data
from skorch.utils import to_tensor
from skorch import NeuralNetClassifier
from process import weights, scaler, cols_inv


#Subclassing the NeuralNetClassifier 
class NeuralNetBase(NeuralNetClassifier):
    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        
        Xi, yi = unpack_data(batch)
        Xinput = to_tensor(Xi, device=self.device)

        y_pred = self.infer(Xinput, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        
        loss_unreduced = self.criterion_(y_pred, y_true)
        loss_reduced = loss_unreduced.mean()
        return loss_reduced
    
    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        
        Xi, yi = unpack_data(batch)
        Xinput = to_tensor(Xi, device=self.device)

        with torch.no_grad():
            y_pred = self.infer(Xinput, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }
    
    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        
        Xi, _ = unpack_data(batch)
        Xinput = to_tensor(Xi, device=self.device)

        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xinput)


