
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

from ignite.engine import Events, create_supervised_trainer,create_supervised_evaluator
import ignite.metrics
import ignite.contrib.handlers
from SwinTransformer import SwinTransformer
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from datautils import get_dls
from datamodule import DataLoaders
import argparse
from ToPatches import ToPatches
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import MeanAbsoluteError, MeanSquaredError

from ignite.engine import Engine
from learner import Learner
from callback.core import *
from callback.tracking import *
from callback.scheduler import *
from callback.patch_mask import *
from callback.transforms import *
from metrics import *


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()
# Dataset and dataloader   
parser.add_argument('--IMAGE_SIZE', type=int, default=32, help='IMAGE_SIZE') # IMAGE_SIZE = 32
parser.add_argument('--NUM_CLASSES', type=int, default=5040, help='NUM_CLASSES') # 5040 2352 1344  672
parser.add_argument('--dset', type=str, default='etth1', help='dataset name')#etth1
parser.add_argument('--context_points', type=int, default=512, help='sequence length') #439 # orig was 336 , WAETHER 7056 27 7056-6912=144  , ELEC 107856, 421 107776- 107856, TRAFFIC 1131, 289632=16     
#PatchTST/64 implies the number of input patches is 64, which uses the look-back window L = 512.
#PatchTST/42 means the number of input patches is 42, which has the default look-back window L = 336.
parser.add_argument('--target_points', type=int, default=720, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')#16 32
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=32, help='patch length') # 16
parser.add_argument('--stride', type=int, default=16, help='stride between patch')# 8
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=0, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model')


args = parser.parse_args()
#print('args:', args)
args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_IMAGE_SIZE'+str(args.IMAGE_SIZE)+'_NUM_CLASSES'+str(args.NUM_CLASSES)  +'_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

#args.save_model_name = f"patchtst_supervised_cw{args.context_points}_tw{args.target_points}_patch{args.patch_len}_stride{args.stride}_epochs{args.n_epochs}_model{args.model_id}"
#args.save_path = f"saved_models/{args.dset}/patchtst_supervised/{args.model_type}/"
#os.makedirs(args.save_path, exist_ok=True)


class PrintShapesCallback(Callback):
    def after_pred(self):
        if self.learn.training:  # Only print shapes during training
            print("y_pred Shape:", self.pred.shape)
            print("y Shape:", self.y.shape)


#class QuantileLoss(nn.Module):
#    def __init__(self, quantile):
#        super(QuantileLoss, self).__init__()
#        self.quantile = quantile
    
#    def forward(self, preds, target):
#        assert 0 < self.quantile < 1
#        errors = target - preds
#        losses = torch.max((self.quantile - 1) * errors, self.quantile * errors)
#        return torch.mean(losses)

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, target):
        errors = target - preds
        is_small_error = torch.abs(errors) < self.delta
        small_error_loss = 0.5 * torch.pow(errors, 2)
        large_error_loss = self.delta * (torch.abs(errors) - 0.5 * self.delta)
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        return torch.mean(loss)

#args = parser.parse_args()

#dls = get_dls(args)


def get_model (c_in, args):
        #c_in=dls.vars
        num_patch=64
        # get number of patches
        #num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
        print('number of patches:', c_in)
        
        model = SwinTransformer(c_in,classes =args.NUM_CLASSES , image_size = args.IMAGE_SIZE,forecast_len=args.target_points
                    ,num_blocks_list=[4,4], dims=[128,128, 256],
                    head_dim=32, patch_size=2, window_size=4,
                    emb_p_drop=0., trans_p_drop=0., head_p_drop=0.2)


        print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

        return model #the function returns the created PatchTST model.

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
        model = get_model(dls.vars,args) #(dls.vars,args)
        # get loss
        #Define the loss function to be used during the model training. Here, mean square error (MSE) is used.
        #loss_func = torch.nn.MSELoss(reduction='mean')
        #loss_func = torch.nn.L1Loss(reduction='mean')
        loss_func=combined_loss


        #quantile = 0.5
        #loss_func = QuantileLoss(quantile)        
        #delta = 1.0
        #loss_func = HuberLoss(delta)
        # get callbacks
        cbs = [RevInCB(dls.vars)] if args.revin else [] # If reversible instance normalization (args.revin) is enabled, use the RevInCB callback. Callbacks are functions that can be hooked into the training loop to provide custom functionalities.
        #cbs += [PatchCB(patch_len=args.IMAGE_SIZE, stride=50)]#Add another callback for handling the patching of time series data.
        cbs += [PatchCB(patch_len=args.patch_len,stride=args.stride )]#
        #define learner context_points
        print('in out', dls.vars, dls.c, dls.len)

        learn = Learner(dls, model, loss_func,cbs=cbs ) #cbs=cbs    #Instantiate the Learner object     ,cbs=cbs               
        # fit the data to the model
        return learn.lr_finder() #Use the learning rate finder method to find the optimal learning rate for training the model and return this value.
  

def train_func(lr=args.lr):
        # get dataloader
        dls = get_dls(args)
        print('in out', dls.vars, dls.c, dls.len)
    
        # get model
        model = get_model(dls.vars, args) # Instantiate the model. ,  (dls.vars, args)
       
        # get loss
        #loss_func = torch.nn.MSELoss(reduction='mean') #Define the loss function.
        #loss_func = torch.nn.L1Loss(reduction='mean')
        loss_func=combined_loss

        #quantile = 0.5
        #delta = 1.0
        #loss_func = HuberLoss(delta)

        #loss_func = QuantileLoss(quantile)  

        # get callbacks
        cbs = [RevInCB(dls.vars)] if args.revin else [] #Set up the RevInCB callback, if reversible instance normalization is enabled.
    
        cbs += [  #PatchCB(patch_len=args.context_points,stride=args.stride)]

        ###            #set up additional callbacks: one for handling patching of the time series data,
                 PatchCB(patch_len=args.patch_len,stride=args.stride),
                    SaveModelCB(monitor='valid_loss', fname=args.save_model_name,      # and another for saving the model whenever the validation loss improves.

                         path=args.save_path ) 


            ]
        cbs += [PrintShapesCallback()]

        # define learner
        learn = Learner(dls, model, 
                            loss_func, 
                            lr=lr, 
                            cbs=cbs,
                            metrics=[mse,mae]
                            )
        print("List of Callbacks:", learn.cbs)  # Add this line here to print the list of callbacks

        # "1cycle" policy, which is a training policy that gradually increases the learning rate and then gradually decreases it. 
        # This has been found to provide good training speed and performance. 
        # The number of epochs is specified by args.n_epochs, 
        # and the maximum learning rate during the cycle is specified by lr.                    
        # fit the data to the model
        learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)

        
def test_func():
    weight_path = args.save_path + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []  #If reversible instance normalization (args.revin) is enabled, use the RevInCB callback. Callbacks are functions that can be hooked into the training loop to provide custom functionalities.
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)#,cbs=cbs
    #This loads the model weights from weight_path, tests the model on the test data dls.test, 
    #and returns the predictions, targets, and the calculated Mean Square Error (mse) and Mean Absolute Error (mae) metrics.
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
    return out

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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
    actual_feature = actual[100, :, feature_idx]
    predicted_feature = predicted[100, :, feature_idx]
    #actual_feature = np.mean(actual[: , : ,feature_idx ], axis=0 )
    #predicted_feature = np.mean(predicted[: , : ,feature_idx ], axis=0)

    # Plot the first sequence
    plt.figure(figsize=(10, 6))
    plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')
    plt.title(f"Actual vs Predicted for Feature {feature_idx}, Sequence 0")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", DEVICE)



    if args.is_train:   # training mode
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:   # testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
    for feature_idx in range(7):  # Assuming there are 7 features
        plot_feature_actual_vs_predicted(out[1], out[0], feature_idx)

   
    print('----------- Complete! -----------')





