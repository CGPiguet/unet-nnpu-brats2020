# Heavily inspired by the Trainer class of the link below
# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch 
import numpy as np 
import os 

class Trainer:
  def __init__(self,
               name             : str,
               model            : torch.nn.Module,
               device           : torch.device,
               criterion        : torch.nn.Module,
               
               optimizer        : torch.optim.Optimizer,
               train_Dataloader : torch.utils.data.Dataset,
               valid_Dataloader : torch.utils.data.Dataset = None, 
               lr_scheduler     : torch.optim.lr_scheduler = None,
               epochs           : int = 40, # 100
               epoch            : int = 0,
               notebook         : bool = False):
  
    self.model            = model
    self.device           = device
    self.criterion        = criterion
    self.optimizer        = optimizer
    self.lr_scheduler     = lr_scheduler
    self.train_Dataloader = train_Dataloader
    self.valid_Dataloader = valid_Dataloader
    self.epochs           = epochs
    self.epoch            = epoch
    self.notebook         = notebook 

    self.train_loss       = []
    self.valid_loss       = []

    self.train_dice_coef  = []
    self.valid_dice_coef  = []

    self.learning_rate    = []

    self.name             = name 

    self.test1            = []
    self.test2            = []

  def run_trainer(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    # progressbar = trange(self.epochs, desc='Progress', leave= False)
    for i in range(self.epochs):
      """Epoch Counter"""
      self.epoch += 1

      """Training block"""
      self._train()

      """Validation block"""
      if self.valid_Dataloader is not None:
        self._validate()

      """Print Status"""
      to_print = 'Epoch: {}/{}\ttrain_loss: {}\ttrain_dice: {}\tvalid_loss: {}\tvalid_dice: {}'
      print(to_print.format(i, self.epochs, self.train_loss, self.train_dice_coef, self.valid_loss, self.valid_dice_coef))

      """Learning rate scheduler block"""
      if self.lr_scheduler is not None:
        if self.valid_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
        else:
            self.lr_scheduler.batch()  # learning rate scheduler step
    # progressbar.close()

  def _train(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    self.model.train() # train mode
    train_losses    = []  # accumulate the losses here
    dice_coefficient = []

    # batch_iter = tqdm(enumerate(self.train_Dataloader), 'Training', total=len(self.train_Dataloader),
    #                   leave= True, position= 0 )

    for i, data in enumerate(self.train_Dataloader):
      input, target = data['img'], data['target']

      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)

      self.optimizer.zero_grad() # Set grad to zero
      
      output  = self.model(input) # One forward pass 
      output  = output.squeeze()

      loss          = self.criterion(output, target) # Calculate loss
      loss_value    = loss.item()
      train_losses.append(loss_value)
      loss.backward() # one backward pass
      self.optimizer.step() # update the parameters

      # Dice Coefficient
      dice_coefficient.append(self._dice_coef(output, target))

      # batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
      # batch_iter.update()

    self.train_loss.append(np.mean(np.array(train_losses)))
    self.train_dice_coef.append(np.mean(dice_coefficient))
    self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

    # batch_iter.close()
    
  def _validate(self):
    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange

    self.model.eval() # evaluation mode
    valid_losses = [] # accumulate the losses here
    dice_coefficient = []

    # batch_iter = tqdm(enumerate(self.valid_Dataloader), 'Validation', total=len(self.valid_Dataloader),
    #                   leave= True, position= 0)
    
    for i, data in enumerate(self.valid_Dataloader):
      input, target = data['img'], data['target']
      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)

      with torch.no_grad():
        output      = self.model(input)
        output      = output.squeeze()
        loss        = self.criterion(output, target)
        loss_value  = loss.item()
        valid_losses.append(loss_value)
        
        # Dice Coefficient
        dice_coefficient.append(self._dice_coef(output, target))

        # batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        # batch_iter.update()
      
    self.valid_loss.append(np.mean(np.array(valid_losses)))
    self.valid_dice_coef.append(np.mean(dice_coefficient))

    # batch_iter.close()

  def plot_loss(self, to_save= False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(self.epochs), self.train_loss)
    plt.plot(np.arange(self.epochs), self.valid_loss)
    plt.legend(['train_loss','valid_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/test loss')

    if to_save:
      name = '_TrainTest_Loss.png'
      name = self.name + name 
      path = 'results'
      
      path = os.path.join(path, name)
      print(path)
      print()
      plt.savefig(path)

  def plot_dice_coefficient(self, to_save= False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(self.epochs), self.train_dice_coef)
    plt.plot(np.arange(self.epochs), self.valid_dice_coef)
    plt.legend(['train_acc','valid_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Train/test Dice Coefficient')

    if to_save:
      name = '_TrainTest_DiceCoef.png'
      name = self.name + name 
      path = '/content/drive/MyDrive/img/UNet/'
      path = os.path.join(path, name)
      plt.savefig(path)

  def _dice_coef(self, output, target, smooth= 1):
    dice_output = torch.where(output>0, 1, 0)
    dice_target = torch.where(target>0, 1, 0)


    intersection =  torch.sum(dice_target*dice_output, dim=(1,2))
    # print(intersection.shape)
    # shape of intercetion = [64]
    union        =  torch.sum(dice_target, dim=(1,2))+ torch.sum(dice_output, dim=(1,2))
    # print(union.shape)
    # shape of union = [64]
    dice         =  torch.mean((2.*intersection + smooth)/(union + smooth), dim=0)
    return dice.item()


