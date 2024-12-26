"""
Alexandra McAdam 
11/10/24

Scoring for GNN training 
"""

from sklearn.metrics import confusion_matrix

#Defining scoring system for each-epoch training
def accept_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return cm[-1][-1]/cm.sum(axis=1)[-1]

def reject_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return 1 - (cm.sum(axis=0)[-1]-cm[-1][-1])/(cm.sum(axis=None)-cm.sum(axis=1)[-1])