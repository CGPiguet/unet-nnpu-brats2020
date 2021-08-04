# Heavily inspired by the Trainer class of the link below
# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
from threading import RLock
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch 
import numpy as np 
import os 
import pandas as pd

from sklearn import metrics

class Trainer:
  def __init__(self,
               name             : str,
               model            : torch.nn.Module,
               device           : torch.device,
               criterion        : torch.nn.Module,              
               optimizer        : torch.optim.Optimizer,
               train_Dataloader : torch.utils.data.Dataset,
               out              : os.path, 
               valid_Dataloader : torch.utils.data.Dataset = None, 
               lr_scheduler     : torch.optim.lr_scheduler = None,
               epochs           : int = 40, # 100
               epoch            : int = 0,
               original_epoch   : int = 0,
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
    self.original_epoch   = original_epoch

    self.out              = out
    self.notebook         = notebook 

    self.train_loss       = []
    self.valid_loss       = []

    self.train_dice_coef  = []
    self.valid_dice_coef  = []

    self.train_ROC        = []
    self.valid_ROC        = []

    self.learning_rate    = []

    self.name             = name 

    self.test1            = []
    self.test2            = []
    
    folder_name = self.out + self.name
    file_name   = 'results.csv'
    save_name   = os.path.join(folder_name, file_name)
    
    if not os.path.isfile(save_name):
      df=pd.DataFrame(columns=["Name","Old","New"])
      df.to_csv(save_name)
      
    



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
      to_print = 'Epoch: {}/{}\ttrain_loss: {}\ttrain_dice_coef: {}\tvalid_loss: {}\tvalid_dice_coef: {}'
      if self.valid_Dataloader is not None:
        print(to_print.format(self.epoch + self.original_epoch, self.epochs + self.original_epoch, self.train_loss[-1], self.train_dice_coef[-1], self.valid_loss[-1], self.valid_dice_coef[-1]))
      else:
        print(to_print.format(self.epoch + self.original_epoch, self.epochs + self.original_epoch, self.train_loss[-1], self.train_dice_coef[-1], None, None))

      """Learning rate scheduler block"""
      if self.lr_scheduler is not None:
        if self.valid_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
        else:
            self.lr_scheduler.batch()  # learning rate scheduler step

      """Save Model""" 
      folder_name = self.out+ self.name
      file_name   = 'epoch_'

      try:
        os.makedirs(folder_name)
      except:
        pass

      torch.save(self.model.state_dict(), os.path.join(folder_name, file_name+ str(self.epoch + self.original_epoch)))      
      
      try:
        os.remove(os.path.join(folder_name, file_name+ str(self.epoch + self.original_epoch - 1)))
      except:
        pass

      if self.epoch % 50 == 0:
        torch.save(self.model.state_dict(), os.path.join(folder_name, file_name+ str(self.epoch + self.original_epoch)+ '_checkpoint'))  

         

    """Percentage of how many time the Negative Risk of nnPU """
    # to_print = self.criterion.number_of_negative_loss, self.criterion.counter, self.criterion.number_of_negative_loss/ self.criterion.counter*100
    # print('# Negative Risk is inferior to beta: {}/{} ({}%)'.format(*to_print))

    # progressbar.close()


  def _train(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    self.model.train() # train mode
    train_losses    = []  # accumulate the losses here
    dice_coefficient= []
    ROC             = []

    saved_output    = []
    saved_target    = []


    # batch_iter = tqdm(enumerate(self.train_Dataloader), 'Training', total=len(self.train_Dataloader),
    #                   leave= True, position= 0 )

    for i, data in enumerate(self.train_Dataloader):
      input, target = data['img'], data['target']
      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)

      self.optimizer.zero_grad() # Set grad to zero
      
      output  = self.model(input) # One forward pass 

      loss, x_grad          = self.criterion(output.squeeze(), target.squeeze()) # Calculate loss
      loss_value    = loss.item()

      train_losses.append(loss_value)
      x_grad.backward() # one backward pass
      self.optimizer.step() # update the parameters

      """Save prediction and target"""
      # saved_output.append(output)
      # saved_target.append(target)

      """Dice Coefficient"""
      dice_coefficient.append(self._dice_coef(output.squeeze(), target.squeeze()))
      
      """ROC"""
      ROC.append(self._ROC_AUC(output.flatten(), target.flatten()))


      # batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
      # batch_iter.update()

    self.train_loss.append(np.mean(np.array(train_losses)))
    self.train_dice_coef.append(np.mean(dice_coefficient))
    self.train_ROC.append(np.mean(ROC))
    # _save_prediction_target( output , target, False)
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
    ROC = []

    # batch_iter = tqdm(enumerate(self.valid_Dataloader), 'Validation', total=len(self.valid_Dataloader),
    #                   leave= True, position= 0)
    
    for i, data in enumerate(self.valid_Dataloader):
      input, target = data['img'], data['target']
      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)
      # input, target = input.squeeze(), target.squeeze()

      with torch.no_grad():
        output      = self.model(input)

        loss, x_grad = self.criterion(output.squeeze(), target.squeeze())
        loss_value  = loss.item()
        valid_losses.append(loss_value)
        
        """Dice Coefficient"""
        dice_coefficient.append(self._dice_coef(output.squeeze(), target.squeeze()))

        """ROC"""
        ROC.append(self._ROC_AUC(output.flatten(), target.flatten()))



        # batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        # batch_iter.update()
      
    self.valid_loss.append(np.mean(np.array(valid_losses)))
    self.valid_dice_coef.append(np.mean(dice_coefficient))
    self.valid_ROC.append(np.mean(ROC))

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
    assert(output.shape == target.shape)
    if target.min() == -1:
      dice_output = torch.sign(output)
      dice_output = torch.where(dice_output>0, 1, 0)
      dice_target = torch.where(target>0, 1, 0)
    elif target.min() == 0:
      dice_output = torch.where(output>0.5, 1, 0)
      dice_target = target


    intersection =  torch.sum(dice_target*dice_output, dim=(1,2))
    # print(intersection.shape)
    # shape of intercetion = [64]
    union        =  torch.sum(dice_target, dim=(1,2))+ torch.sum(dice_output, dim=(1,2))
    # print(union.shape)
    # shape of union = [64]
    dice         =  torch.mean((2.*intersection + smooth)/(union + smooth), dim=0)
    return dice.item()

  def _ROC_AUC(self, output, target):
    roc_target = target.cpu().numpy()
    roc_output = torch.tanh(output)
    roc_output = roc_output.detach().cpu().numpy()


    fpr, tpr, thresholds = metrics.roc_curve(roc_target, roc_output)
    ROC = metrics.auc(fpr, tpr)
    
    return ROC

  def _save_results(self):
      if self.valid_Dataloader is not None:
          # print(len(self.train_loss), len(self.train_dice_coef), len(self.valid_loss),len(self.valid_dice_coef))
          df = pd.DataFrame({
              'train_loss': self.train_loss[-1],
              'train_dice': self.train_dice_coef[-1],
              'train_ROC_AUC': self.train_ROC[-1],
              'valid_loss': self.valid_loss[-1],
              'valid_dice': self.valid_dice_coef[-1],
              'valid_ROC_AUC': self.valid_ROC[-1]
          })
      else :
          # print(len(self.train_loss), len(self.train_dice_coef))
          df = pd.DataFrame({
              'train_loss': self.train_loss[-1],
              'train_dice': self.train_dice_coef[-1],
              'train_ROC_AUC': self.train_ROC[-1]
          })

      folder_name = self.out + self.name
      file_name   = 'results.csv'
      save_name   = os.path.join(folder_name, file_name)
      
      with open(save_name, 'a') as f:
        if not os.path.isfile(save_name):   
          df.to_csv(f, header=True)
        else:
          df.to_csv(f, header=False)
      

  # # def _save_prediction_target(self, output , target, valid):
  #   save_output = output.detach().cpu()
  #   save_target = target.cpu()

  #   folder_name = self.out+ self.name
  #   if not valid:
  #     prediction_folder = os.path.join(folder_name, 'train_Prediction')
  #     target_folder     = os.path.join(folder_name, 'train_GroundTruth')
  #   else:
  #     prediction_folder = os.path.join(folder_name, 'valid_Prediction')
  #     target_folder     = os.path.join(folder_name, 'valid_GroundTruth')

  #   prediction_name   = 'pred' + self.epoch + '.pt'
  #   target_name       = 'target'+ self.epoch + '.pt'

  #   try:
  #     os.makedirs(prediction_folder)
  #   except:
  #     pass
  #   try:
  #     os.makedirs(target_folder)
  #   except:
  #     pass
  #   torch.save(save_output, os.path.join(prediction_folder, prediction_name))




    


