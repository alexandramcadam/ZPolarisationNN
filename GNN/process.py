"""
Alexandra McAdam 
11/10/24

Process Monte Carlo sim data in .h5 file into graphs for input to GNN 
"""

import h5py
import numpy as np
import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Dataset
from collections.abc import Sequence
from skorch.utils import to_numpy 


#file directories
hfivesdir =  "../datasets/new_Input_Polarisation_16st_October2024.h5" #absolute path to dataset 
graphsdir = "../graphdata/new_Input_Polarisation_16st_October2024.h5" #directory to store graph data 


#subclass for processing the dataset
#initially used to convert dataset into graphs and then used to access the graphs for training/evaluation
class MyOwnDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return 'new_Input_Tagging_v1.h5' 
    
    @property
    def processed_file_names(self):
        return 'new_Input_Tagging_v1.pt' 
    
    def download(self):
        pass

    #uncomment the commented lines for converting h5 dataset into graphs
    #comment it again to save time when only accessing the graphs
    def process(self):
        with h5py.File(hfivesdir, 'r') as f:
            #relabel truth labels - 0 for longitudinal, 1 for transverse 
            df_ones = pd.DataFrame(f['LargeRJet']['1d'][:])
            df_ones.replace({'polarisation_type':{1:0, 2:1}}, inplace=True)  
            
            self.ones = df_ones #graph level information
            self.data = pd.DataFrame(f['LargeRJet']['2d']) #2d data has constituent info 
            

        pass #comment out the below code once data is processed into graphs to avoid unneccesary computation
        # i = 0
        # for row in self.data.itertuples():
        #    #Obtaining the features of each graph node/constituent
        #    node_feats = self._get_node_features(row, i)
        
        #    #Obtaining the truth label and other various graph level information
        #    polarisation_type = self._get_labels(i)
        
        #    #Constructing the graph dataset 
        #    data = Data(x = node_feats, y = polarisation_type) #y is the truth label 
        
        #    #Saving the graph dataset
        #    torch.save(data, osp.join(self.processed_dir, f'new_Input_Tagging_{i}.pt'))
        
        #    #Lines for printing number of iterations, only used to monitor
        #    #the graph preprocessing
        #    if (i%5000==1):
        #        print(i)
        #    i += 1

    
    #function to obtain the features of each graph node
    def _get_node_features(self, row, i):
        all_node_feats = []
        
        #each row contains 7 items of information, starting from row 1 avoids header 
        for column in row[1:]: 

            #check not Nan 
            if np.isnan(column[0]) == 0: #might throw error if not numeric (should be numeric)
                nodes = self._preprocess(column, i) #adds processed values to node features 
                all_node_feats.append(nodes)           
                
        #convert graph nodes into array
        all_node_feats = np.asarray(all_node_feats)
        
        return torch.tensor(all_node_feats, dtype=torch.float)
    
    #function to process the raw track/calo-level information to final graph features input
    #important to know which dataset column corresponds to which feature 
    def _preprocess(self, column, i):

        E_const = column[6] * np.cosh(column[3]) #energy of constituent  pT*cosh(eta)
        log_E_const = np.log(E_const)
        log_pT = np.log(column[6]) #log particle transverse momentum

        #check if the constituent is a lepton (isLepton==1) or a jet
        if column[4] == 1:
            p_ref, m_ref, eta_ref, phi_ref = (self.ones['Vlep_pT'][i], self.ones['Vlep_mass'][i], self.ones['Vlep_eta'][i], self.ones['Vlep_phi'][i])
        else:
            p_ref, m_ref, eta_ref, phi_ref = (self.ones['FJ_pT'][i], self.ones['FJ_mass'][i], self.ones['FJ_eta'][i], self.ones['FJ_phi'][i])


        E_ref = np.sqrt(p_ref ** 2 + m_ref ** 2) #energy lepton/jet 
        rel_pT = np.log(column[6] / p_ref) #log(pT/pT(lepton/jet))
        delta_eta = column[3] - eta_ref #difference in eta between constituent and lepton/jet
        delta_phi = column[5] - phi_ref #difference in phi 
        R_const = np.sqrt(delta_eta ** 2 + delta_phi ** 2) #constituent radial distance from lepton/jet axis in eta-phi plane 

        #same as ParticleNet input features
        nodes = [delta_eta, delta_phi, log_pT, log_E_const, rel_pT, np.log(E_const / E_ref), R_const]

        #add isLepton, charge
        nodes.extend([column[4], column[2]]) 

        return nodes

    #function to obtain truth label 
    def _get_labels(self, i):
        p_type = np.asarray(self.ones['polarisation_type'][i])

        return torch.tensor(p_type, dtype=torch.int64)

    def len(self):
        return self.ones.shape[0]

    #retrieves datapoint at index idx 
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'new_Input_Tagging_{idx}.pt'), weights_only=False) 
        return data

with h5py.File(hfivesdir) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

#constructing the class-weight
df.replace({'polarisation_type':{1:0, 2:1}}, inplace=True)  
value_counts = df['polarisation_type'].value_counts()
total_count = len(df)
weights = total_count / value_counts.sort_index().values
weights = torch.tensor(weights, dtype=torch.float)

#function for ZZ invariant mass 
def invariant_mass(FJ_pT, FJ_eta, FJ_phi, Vlep_pT, Vlep_eta, Vlep_phi):
    #four-momentum components for FJ boson
    FJ_px = FJ_pT * np.cos(FJ_phi)
    FJ_py = FJ_pT * np.sin(FJ_phi)
    FJ_pz = FJ_pT * np.sinh(FJ_eta)
    FJ_E = df['FJ_pT']*np.cosh(df['FJ_eta']) #for high energies 

    #four-momentum components for Vlep boson
    Vlep_px = Vlep_pT * np.cos(Vlep_phi)
    Vlep_py = Vlep_pT * np.sin(Vlep_phi)
    Vlep_pz = Vlep_pT * np.sinh(Vlep_eta)
    Vlep_E = df['Vlep_pT']*np.cosh(df['Vlep_eta']) 

    #total four-momentum
    total_E = FJ_E + Vlep_E
    total_px = FJ_px + Vlep_px
    total_py = FJ_py + Vlep_py
    total_pz = FJ_pz + Vlep_pz
    
    #invariant mass
    invariant_mass = np.sqrt(total_E**2 - (total_px**2 + total_py**2 + total_pz**2))
    return invariant_mass

inv_mass = invariant_mass(df['FJ_pT'],df['FJ_eta'],df['FJ_phi'], df['Vlep_pT'],df['Vlep_eta'],df['Vlep_phi'])


#call for processing/accessing dataset, the graphsdir is where the dataset is saved
dataset = MyOwnDataset(root=graphsdir)




#shuffling the graphs and dividing into training and testing dataset
#seed is fixed as 1234
torch.manual_seed(1234)
dataset = dataset.shuffle()
a = int(len(dataset))
b = int(a*0.9)
train_dataset = dataset[:b]
test_dataset = dataset[b:a]

test_inv_mass = inv_mass[b:a]





#Class for treating the graphs dataset as numpy array
#Convenient for converting graph-level information into array
class SliceDataset(Sequence):
    def __init__(self, dataset, idx=0, indices=None):
        self.dataset = dataset
        self.idx = idx
        self.indices = indices

        self.indices_ = (self.indices if self.indices is not None
                         else np.arange(len(self.dataset)))
        self.ndim = 1

    def __len__(self):
        return len(self.indices_)

    @property
    def shape(self):
        return (len(self),)

    def transform(self, data):
        """Additional transformations on ``data``.

        Note: If you use this in conjuction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``data`` is a single rows.

        """
        return data

    def _select_item(self, Xn):
        # Raise a custom error message when accessing out of
        # bounds. However, this will only trigger as soon as this is
        # indexed by an integer.
        try:
            if (self.idx == 0):
                return Xn.x
            if (self.idx == 1):
                return Xn.y
            if (self.idx == 2):
                return Xn.m
        except IndexError:
            name = self.__class__.__name__
            msg = ("{} is trying to access element {} but there are only "
                   "{} elements.".format(name, self.idx, len(Xn)))
            raise IndexError(msg)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xn = self.dataset[self.indices_[i]]
            Xi = self._select_item(Xn)
            return self.transform(Xi)

        cls = type(self)
        if isinstance(i, slice):
            return cls(self.dataset, idx=self.idx, indices=self.indices_[i])

        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError("SliceDataset only supports slicing with 1 "
                                 "dimensional arrays, got {} dimensions instead."
                                 "".format(i.ndim))
            if i.dtype == bool:
                i = np.flatnonzero(i)

        return cls(self.dataset, idx=self.idx, indices=self.indices_[i])

    def __array__(self, dtype=None):
        # This method is invoked when calling np.asarray(X)
        # https://numpy.org/devdocs/user/basics.dispatch.html
        X = [self[i] for i in range(len(self))]
        if np.isscalar(X[0]):
            return np.asarray(X)
        return np.asarray([to_numpy(x) for x in X], dtype=dtype)