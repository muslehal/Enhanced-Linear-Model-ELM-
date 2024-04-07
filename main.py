
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
from learner import Learner
from callback.core import *
from callback.tracking import *
from callback.scheduler import *
from callback.patch_mask import *
from callback.transforms import *
from metrics import *
from datautils import get_dls
from ELM import ELM
import time
import json
import random
import argparse
import datetime
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.cuda.amp import GradScaler, autocast

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--model_name2', type=str, default='ELM', help='model_name2')
# IntegratedModel   model1 model2 dlinear
parser.add_argument('--dset', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')

parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')

# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

# Optimization args
parser.add_argument('--n_epochs', type=int, default=3, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model')


#args, unparsed = parser.parse_known_args()

#config = (args)


args = parser.parse_args()
#print('args:', args)

args.save_model_name = str(args.model_name2)+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset 
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


def get_model(c_in, args):

    """
    c_in: number of input variables
    """
     
    #model_type ='IntegratedModel'#IntegratedModel
    # Select model based on model_type parameter
    if args.model_name2 == 'ELM':
        model = ELM(
        pred_len=args.target_points,
            seq_len =args.context_points,
 batch_size2=args.batch_size ,
            enc_in=c_in,
             
                ) 
        
       
    
    else:
        raise NotImplementedError(f"Unknown model: {args.model_name2}")

    return  model


def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)

def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    #loss_func = torch.nn.L1Loss(reduction='mean')
    loss_func=combined_loss
  
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    # define learner
    learn = Learner(dls,model,  loss_func , cbs=cbs )                      
    # fit the data to the model
    return learn.lr_finder()


#def train_func(model_type,lr=args.lr):
def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    #print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(dls.vars, args)
    #model = get_model(dls.vars, args, model_type)

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    #loss_func = torch.nn.L1Loss(reduction='mean')
    loss_func=combined_loss
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model , loss_func,
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse,mae]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path =args.save_path+'/' + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         
    return out



def plot_feature_actual_vs_predicted(actual, predicted, feature_idx):
    """
    Plot the actual vs predicted values for a specific feature for the first sequence.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    """
   
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Select the first sequence for the given feature index
    actual_feature = actual[0, :, feature_idx]
    predicted_feature = predicted[0, :, feature_idx]

    # Plot the first sequence
    plt.figure(figsize=(10, 6))
    plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')

    # Set title with larger font size
    plt.title("Actual vs Predicted", fontsize=16)

    # Set legend with larger font size
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    if args.is_train:   # training mode
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:   # testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
        for feature_idx in range(7):  # Assuming there are 7 features
            plot_feature_actual_vs_predicted(out[0], out[1], feature_idx)

   
    print('----------- Complete! -----------')


