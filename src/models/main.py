# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:32:24 2021

@author: bjorn

main script to run all training/eval scripts from
"""

import copy
import wandb
import torch
import torch.nn as nn
import time
import math
import numpy as np

from data_loader import h5py_loader, torch_data_loader
from TransformerModel import TransformerModel
from train_eval import train, evaluate

# load data
af_data_path = '/af_full.h5'
normal_data_path = '/normal_full.h5'
af_array = h5py_loader(af_data_path)
normal_array = h5py_loader(normal_data_path)
df1 = torch_data_loader(normal_array)
df2 = torch_data_loader(af_array) 
df = df1.append(df2) # combine to one df
Y1 = [1]*len(df1) # Normal
Y2 = [0]*len(df2) # AF
Y = []
Y.extend(Y1)
Y.extend(Y2)
# del df1, df2, Y1, Y2 # delete individual dataframes if needed
train_loader, val_loader = torch_data_loader(df, Y, train_size=0.85, batch_size=10, random_state=1)

# WandB - Initialize run
wandb.init(entity="user_name", project="project_name")
# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 10          # input batch size for training (default: 64)
# config.test_batch_size = batch_size    # input batch size for testing (default: 1000)
config.epochs = 100             # number of epochs to train (default: 10)
config.lr = 0.0001               # learning rate (default: 0.01)
# config.momentum = 0.1          # SGD momentum (default: 0.5) 
# config.seed = 42               # random seed (default: 42)
config.log_interval = 1     # how many batches to wait before logging training status
config.emsize = 64 # embedding dimension == d_model
config.dim_feedforward = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
config.nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
config.nhead = 4 # the number of heads in the multiheadattention models
config.n_conv_layers = 2 # number of convolutional layers (before transformer encoder)
config.dropout = 0.25 # the dropout value
config.dropout_other = 0.1 # dropout value for feedforward output layers
config.n_class = 2
config.use_rri_features = True         # using raw signal + rri features during training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set to gpu if possible
def main():
  model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
  # criterion = nn.CrossEntropyLoss()
  criterion = nn.BCEWithLogitsLoss() # pass logits as input (not probabilities)
  # criterion = nn.BCELoss() # pass probabilities (using sigmoid) as input
  best_val_loss = float("inf")
  epochs = config.epochs # The number of epochs
  best_model = None
  lr = config.lr # learning rate
  # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98)) #weight_decay=1e-6
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)
  # WandB – "wandb.watch()" automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
  wandb.watch(model, log="all")
  history = dict(train=[], val=[], val_acc=[], val_sens=[], val_spec=[])
  for epoch in range(1, epochs + 1):
      epoch_start_time = time.time()
      model, train_loss = train(config, model, optimizer, criterion)
      val_loss, cm, val_acc, val_sens, val_spec = evaluate(args=config, eval_model=model, data_source=val_loader, criterion=criterion)
      epoch_time = time.time() - epoch_start_time
      # history['train'].append(train_loss)
      history['val'].append(val_loss)
      history['val_acc'].append(val_acc)
      history['val_sens'].append(val_sens)
      history['val_spec'].append(val_spec)
      wandb.log({"Epoch Time [s]": epoch_time}) # log time for epoch
      # table_sens = wandb.Table(data=history['val_sens'], columns=["Sensitivity"])
      # wandb.log({"Test Sensitivity" : wandb.plot.line(table_sens, "Sensitivity", title="Sensitivity over epochs")})
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, epoch_time,
                                      val_loss, math.exp(val_loss)))
      print('-' * 89)
      print('valid conf mat:\n', cm)
      print('valid accuracy:', np.round(val_acc, 4))
      print('-' * 89)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
      if epoch > 30:
        scheduler.step()  

  return best_model        

if __name__ == '__main__':
  best_model = main()     


# save model 
best_model_wts = copy.deepcopy(best_model.state_dict())
torch.save(best_model, 'model_save_path.pth')

