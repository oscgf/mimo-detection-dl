# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:33:13 2021

@author: Óscar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import is_tensor

class Net(nn.Module):
    def __init__(self,M,n):
        super().__init__()
        self.fc1 = nn.Linear(n, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, M**n)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
                
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
    
    def restart(self):
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.normal_(std=0.01)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.normal_(std=0.01)
        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data.normal_(std=0.01)
        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data.normal_(std=0.01)

    
class SymbolsDataset(Dataset):
    
    def __init__(self, Rn, Cn, Bn):
        self.Rn = Rn    # Symbols
        self.Cn = Cn    # Combinations
        self.Bn = Bn    # Bits

    def __len__(self):
        return len(self.Rn.T)
     
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        
        symbol = self.Rn[:,idx]
        comb = self.Cn[idx]
        
        k = int(len(self.Bn.T) / len(self.Rn.T))
        bits = self.Bn[:,k*idx:k*idx+k]
        
        return [symbol,comb,bits]
    
    def split_data(self, split):
        Nb = len(self.Bn.T)
        Ns = len(self.Rn.T)
        k = int(Nb/Ns)
        
        train_Rn, valid_Rn = torch.split(self.Rn, [Ns-split,split],dim=1)
        train_Cn, valid_Cn = torch.split(self.Cn, [Ns-split,split])
        train_Bn, valid_Bn = torch.split(self.Bn, [Nb-k*split,k*split],dim=1)

        return [SymbolsDataset(train_Rn,train_Cn,train_Bn),SymbolsDataset(valid_Rn,valid_Cn,valid_Bn)]
