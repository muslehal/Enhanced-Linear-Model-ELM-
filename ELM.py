
from math import fabs
from pickle import TRUE
from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
from torch.cuda.amp import autocast
#from callback.revin import RevIN



from torch import nn
import torch


class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, pred_len, seq_len, enc_in, batch_size2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size2=batch_size2

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        
        self.individual =False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
                               
            
    def forward(self, x):
        

            if self.individual:
                output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
            else:
                x=self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
                #x=F.gelu(self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1))
            # print(x.shape)
            
            x=x.permute(0,2,1)
            x=self.batch_norm(x)
            
            x=x.permute(0,2,1)
            return x



class Model2(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, pred_len,seq_len,enc_in):
        super(Model2, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        self.individual=False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                #self.batch_norm(self.Linear.append(nn.Linear(self.seq_len,self.pred_len)))
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):
        
            # x: [Batch, Input length, Channel]
            seq_last = x[:,-1:,:].detach()

            x = x - seq_last
            if self.individual:
                output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](x[:,:,i])
                x = output
            else:
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            x = x + seq_last

            
            x=x.permute(0,2,1)
            x=self.batch_norm(x)
            x =x.permute(0,2,1)
        
            return x # [Batch, Output length, Channel]



class ELM(nn.Module):
    def __init__(self, pred_len, seq_len, enc_in, batch_size2):
        super(ELM, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        
        self.model2 = Model2(pred_len, seq_len, enc_in)
        self.model = Model(pred_len, seq_len, enc_in, batch_size2)

    def forward(self, x):
        # Get outputs from both models
        output_model2 = self.model2(x)
        output_model = self.model(x)
       
        # Average the outputs instead of concatenating
        #x =(output_model2 + output_model) /2
      
        x = F.gelu((output_model2 + output_model) /2)
        #x = F.silu((output_model2 + output_model) /2)
        #x = torch.mean(torch.stack((output_model2, output_model), dim=0), dim=0)

        
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        return x


