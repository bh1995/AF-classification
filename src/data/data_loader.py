# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:08:43 2021

@author: bjorn

Script for loading the h5py file format
"""

import h5py
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def h5py_loader(path, driver):
    h5f = h5py.File(path,'r')
    array = h5f[driver][:]
    h5f.close()
    df = pd.DataFrame(data=array)
    df.dropna(inplace=True)
    return df



def torch_data_loader(df, Y, train_size=0.85, batch_size=10, random_state=1):
    # Perform shuffling on data
    print('----------------------- Before shuffle -----------------------')
    print(df.head())
    print(df.shape)
    print(Y)
    print(len(Y))
    # shuffle data 
    df['labels'] = Y
    df = df.sample(frac=1, random_state=random_state) # set random seed for reproducability
    Y = list(df['labels'])
    df = df.drop(['labels'], axis=1)
    print('----------------------- After shuffle -----------------------')
    print(df.head())
    print(df.shape)
    print(Y)
    print(len(Y))
    
    n = train_size # n=0.7
    t1 = int(len(df)*(n/1))
    x_train = torch.tensor(df.iloc[0:t1].values)
    # x_train_rr = torch.tensor(df_rr.iloc[0:t1].values)
    x_train = x_train[:, :, None]
    n = 1.0
    t2 = int(len(df)*(n/1))
    x_val = torch.tensor(df.iloc[t1:t2].values)
    # x_val_rr = torch.tensor(df_rr.iloc[t1:t2].values)
    x_val = x_val[:, :, None]
    n = 1
    t3 = int(len(df)*(n/1))
    x_test = torch.tensor(df.iloc[t2:t3].values)
    # x_test_rr = torch.tensor(df_rr.iloc[t2:t3].values)
    x_test = x_test[:, :, None]
    
    y_train = torch.tensor(Y[0:t1])
    y_val = torch.Tensor(Y[t1:t2])
    y_test = torch.Tensor(Y[t2:t3])
    
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print('df and Y same length:', len(df)==len(Y))
    print('Train set len:', len(train_ds))
    print('Val set len:', len(val_ds))
    
    return train_loader, val_loader



