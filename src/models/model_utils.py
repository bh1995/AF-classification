# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:23:12 2021

@author: bjorn

script holding for all ulility functions
"""
import numpy as np
import torch
import torch.nn as nn
import math
from ecgdetectors import Detectors
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(1), :].squeeze(1)
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


def get_rri(sig, fs=300, n_beats=10):
  """
  Input: 10s ECG signal (torch tensor)
  Output: RRI, time in between each beat (10 beats otherwise 0-padded)
  """
  detectors = Detectors(fs)
  sig_n = sig.numpy()  
  # if type(sig).__module__ != np.__name__:
  #  sig_n = sig.numpy()
  #else:
   # sig_n = sig
  # print(sig_n.shape)
  rri_list = []
  for i in range(sig_n.shape[0]):
    r_peaks = detectors.pan_tompkins_detector(sig_n[i])
    rri = np.true_divide(np.diff(r_peaks), fs)
    if len(rri) < n_beats:
      rri = np.pad(rri, (0, n_beats-len(rri)), 'constant', constant_values=(0))
    if len(rri) > n_beats:
      rri = rri[0:n_beats]
    rri_list.append(rri)
  
  rri_stack = np.stack(rri_list, axis=0)
  # print(rri_stack.shape) 
  return rri_stack  

from matplotlib.lines import Line2D 
def plot_grad_flow(named_parameters):
  """
  input: model.name_parameters()
  note: put this function after loss.backward()
  """
  ave_grads = []
  max_grads = []
  layers = []
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
        layers.append(n)
        if p.grad is None: continue
          # ave_grads.append(0)
          # max_grads.append(0)
        else:
          ave_grads.append(p.grad.abs().mean())
          max_grads.append(p.grad.abs().max())
  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  plt.legend([Line2D([0], [0], color="c", lw=4),
              Line2D([0], [0], color="b", lw=4),
              Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
  plt.show() 


