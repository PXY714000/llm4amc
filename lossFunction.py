import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def nmseCalculation(sinr_hat,sinr):
    mse = torch.mean((sinr_hat-sinr)**2)
    meanSinrPower = torch.mean(sinr ** 2)
    nmse = mse / meanSinrPower
    return nmse


class nmseLoss(nn.Module):
    def __init__(self):
        super(nmseLoss,self).__init__()
    
    def forward(self,sinr_hat,sinr):
        nmse = nmseCalculation(sinr_hat,sinr)
        return nmse
    

def mseCalculation(sinr_hat,sinr):
    mse = torch.mean((sinr_hat-sinr)**2)
    return mse


class mseLoss(nn.Module):
    def __init__(self):
        super(mseLoss,self).__init__()
    
    def forward(self,sinr_hat,sinr):
        nmse = mseCalculation(sinr_hat,sinr)
        return nmse