import sys
import argparse



from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from sklearn import metrics

from preprocess2D import *
from preprocess3D import *
from dataset2D import *
from dataset3D import *
from model import *
from nnPULoss import *
from trainer import *
from utils import *
%matplotlib inline
%load_ext autoreload
%autoreload 2


from typing import Tuple


class PULossCustom(nn.Module):
  def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), beta= 0,gamma= 1, nnPU= True)-> None:
    """Implementation of the non-negative risk estimator from Positive-Unlabeled Learning with
Non-Negative Risk Estimator 

    Args:
        prior (float): The positive class prior
        loss (lambda): Loss function used in the risks estimators. Defaults to (lambda x: torch.sigmoid(-x)).
        beta (int): Beta parameter from the nnPU paper. Defaults to 0.
        gamma (int): Gamma parameter from the nnPU paper. Defaults to 1.
        nnPU (bool): If set to True, apply the non-negative risk estimator. If set to False, apply the unbiased risk estirmator. Defaults to True.

    Raises:
        NotImplementedError: The prior should always be set between 0 and 1 (0,1)
    """
    super(PULossCustom,self).__init__()
    if not 0 < prior < 1:
      raise NotImplementedError("The class prior should be in (0,1)")
    self.prior = prior
    self.beta  = beta
    self.gamma = gamma
    self.loss  = loss
    self.nnPU  = nnPU
    self.positive = 1
    self.negative = -1
    self.min_count = torch.tensor(1.)
    self.number_of_negative_loss = 0
    self.counter = 0

  def forward(self, input, target) -> Tuple[torch.tensor, torch.tensor]:
    """Forward pass of the loss function with non-negative risk estimator

    Args:
        input (torch.tensor): Prediction of the model for the data
        target (torch.tensor): Ground-truth of the data

    Returns:
        (output (torch.tensor), x_grad torch.tensor)): Returns the results of the non-negative risk estimator, and the value used in the backward propagation
    """
    input, target = input.view(-1), target.view(-1)
    assert(input.shape == target.shape)
    positive, unlabeled = target == self.positive, target == self.negative
    positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float) 
   
    if input.is_cuda:
      self.min_count = self.min_count.cuda()
      self.prior = self.prior.cuda()

    n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))

    # Loss function for positive and unlabeled
    ## All loss functions are unary, such that l(t,y) = l(z) with z = ty
    y_positive  = self.loss(input).view(-1)  # l(t, 1) = l(input, 1)  = l(input * 1)
    y_unlabeled = self.loss(-input).view(-1) # l(t,-1) = l(input, -1) = l(input * -1)
    
    # # # Risk computation
    positive_risk     = torch.sum(y_positive  * positive  / n_positive)
    positive_risk_neg = torch.sum(y_unlabeled * positive  / n_positive)
    unlabeled_risk    = torch.sum(y_unlabeled * unlabeled / n_unlabeled)
    negative_risk     = unlabeled_risk - self.prior * positive_risk_neg
    print('Positive Risk: {}'.format(positive_risk* self.prior))
    print('Positive Risk neg: {}'.format(positive_risk_neg*self.prior))
    print('Unlabeled Risk : {}'.format(unlabeled_risk))
    print('Negative Risk : {}'.format(negative_risk))
    # Update Gradient 
    if negative_risk < -self.beta and self.nnPU:
      # Can't understand why they put minus self.beta
      output = self.prior * positive_risk - self.beta
      x_grad =  - self.gamma * negative_risk  
      self.number_of_negative_loss += 1
    else:
      # Rpu = pi_p * Rp + max{0, Rn} = pi_p * Rp + Rn
      output = self.prior * positive_risk + negative_risk
      x_grad = self.prior * positive_risk + negative_risk
    self.counter += 1

    return output, x_grad, positive_risk, positive_risk_neg, unlabeled_risk


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_data, valid_data = preprocess_brats2020_2D('T1',root_dir='MICCAI_BraTS2020_TrainingData/', ratio_train_valid= 0.8, ratio_P_to_U= 0.95)
train_dataloader, valid_dataloader,_ = select_dataloader(True, train_data, valid_data, "nnPULoss", 8, is_validation= True, num_worker= 2, prior= 0.5)


model  = unet().to(device)
path = 'model_saved_nnPULoss Prior=0.5 lr=0.001/epoch_200_checkpoint.pth'
# path = 'model_saved_NoPrior/epoch_200_checkpoint.pth'
model.load_state_dict(torch.load(path ,map_location=torch.device('cpu'))['model_state_dict'])
optimizer =  torch.optim.SGD(model.parameters(), lr = 0.001,  weight_decay=0.005)
optimizer.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['optimizer_state_dict'])



loss_fn = PULossCustom(0.5)
train_pPlus_risk = []
train_pMinus_risk = []
train_uMinus_risk = []
train_loss = []

valid_pPlus_risk = []
valid_pMinus_risk = []
valid_uMinus_risk = []
valid_loss = []
for i in range(1):
    for data in train_dataloader:
        input, target = data['img'], data['target']
        input, target = input.to(device), target.to(device) # Send to device (GPU or CPU)
        optimizer.zero_grad() # Set grad to zero
      
        output  = model(input) # One forward pass 

        loss, x_grad, positive_risk, positive_risk_neg, unlabeled_risk  = loss_fn(output.squeeze(), target.squeeze()) # Calculate loss
        train_pPlus_risk.append(positive_risk)
        train_pMinus_risk.append(positive_risk_neg)
        train_uMinus_risk.append(unlabeled_risk)
        
        loss_value    = loss.item()

        train_loss.append(loss_value)
        x_grad.backward() # one backward pass
        optimizer.step() # update the parameters
        
    for data in valid_dataloader:
        input, target = data['img'], data['target']
        input, target = input.to(device), target.to(device) # Send to device (GPU or CPU)
        output  = model(input) # One forward pass 
        
        loss, x_grad, positive_risk, positive_risk_neg, unlabeled_risk  = loss_fn(output.squeeze(), target.squeeze()) # Calculate loss
        valid_pPlus_risk.append(positive_risk)
        valid_pMinus_risk.append(positive_risk_neg)
        valid_uMinus_risk.append(unlabeled_risk)
        
        
        loss_value    = loss.item()
        valid_loss.append(loss_value)
    

df = pd.DataFrame({
    'train_pPlus_risk': train_pPlus_risk,
    'train_pMinus_risk': train_pMinus_risk,
    'train_uMinus_risk': train_uMinus_risk,
    'train_loss': train_loss,
    'valid_pPlus_risk': valid_pPlus_risk,
    'valid_pMinus_risk': valid_pMinus_risk, 
    'valid_uMinus_risk': valid_uMinus_risk, 
    'valid_loss': valid_loss   
})


folder_name = 'RiskAnalysis'
file_name   = 'RiskResults.csv'
save_name   = os.path.join(folder_name, file_name)

with open(save_name, 'a') as f:
    df.to_csv(f, header=f.tell()==0)
    