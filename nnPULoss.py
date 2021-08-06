import torch 
import torch.nn as nn 

from typing import Tuple


class PULoss(nn.Module):
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
    super(PULoss,self).__init__()
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

    return output, x_grad 