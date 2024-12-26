"""
Alexandra McAdam 
16/11/24

Scoring for DNN training 
"""

from sklearn.metrics import confusion_matrix, roc_curve, auc
from process import weights


#Defining scoring system for each-epoch training
def auc_epoch(net, X, y):
    y_score = net.predict_proba(X)
    fpr, tpr, threshold = roc_curve((y == len(weights)-1), y_score[:,-1])
    auc_value = auc(fpr, tpr)
    
    return auc_value

def accept_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return cm[-1][-1]/cm.sum(axis=1)[-1]

def reject_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return 1 - (cm.sum(axis=0)[-1]-cm[-1][-1])/(cm.sum(axis=None)-cm.sum(axis=1)[-1])