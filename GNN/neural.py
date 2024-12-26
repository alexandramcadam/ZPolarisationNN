"""
Alexandra McAdam 
11/10/24

Define GNN based on NeuralNetClassifier
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from skorch.dataset import get_len
from skorch.utils import to_tensor 
from skorch import NeuralNetClassifier
from process import SliceDataset, weights


#subclassing NeuralNetClassifier to make it compatible for graph training
class NeuralNetGraph(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #use torch geometric dataloader as default dataloader    
        self.iterator_train = DataLoader
        self.iterator_valid = DataLoader
    
    #modify loss function 
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        
        loss_unreduced = self.criterion_(y_pred, y_true)
        loss = (loss_unreduced).mean()

        return loss

    #modify train step to accommodate graphs 
    def train_step(self, batch, **fit_params):
        self.module_.train()
        
        inputs = batch.to(self.device)
        labels = batch.y.to(self.device)
        
        self.optimizer_.zero_grad()
        out = self.module_(inputs)
        
        loss = self.get_loss(out, labels, inputs)
        
        loss.backward()
        self.optimizer_.step()
        
        return {'loss' : loss, 'y_pred' : out}
    
    #modify validation step to acommodate graphs
    def validation_step(self, batch, **fit_params):
        inputs = batch.to(self.device)
        labels = batch.y.to(self.device)

        with torch.no_grad():
            out = self.module_(inputs)
            loss = self.get_loss(out, labels, inputs)
        return {
            'loss': loss,
            'y_pred': out,
        }
    
    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        inputs = batch.to(self.device)
        
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.module_(inputs)
    
    def run_single_epoch(self, iterator, training, prefix, 
                        step_fn, **fit_params):
        if iterator is None:
            return

        batch_count = 0
        for batch in iterator:
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (get_len(batch[0]) 
                          if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=[batch.x, batch.y], 
                        training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        self.check_training_readiness()
        epochs = epochs if epochs is not None else self.max_epochs

        #instead of using X, y for training features and label, use them for training features and validation features
        #the label is instead extracted in train_step, validation_step, etc.
        dataset_train, dataset_valid = X, y
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }
        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = None
        if dataset_valid is not None:
            iterator_valid = self.get_iterator(dataset_valid, training=False)

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(iterator_train, training=True, 
                                  prefix="train",
                                  step_fn=self.train_step, **fit_params)

            self.run_single_epoch(iterator_valid, training=False, 
                                  prefix="valid",
                                  step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self    

