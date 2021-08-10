import sys
import argparse
import os

import torch 
import torch.nn as nn
import numpy as np
from utils import *
from model import *



def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Determine ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument('--rootdir', '-r', type=str, default="MICCAI_BraTS2020_TrainingData",
                        help='Root directory of the BraTS2020')
    parser.add_argument('--device', '-d', type=torch.device, default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Determine the torch device, AutoDetection per default')
    
    parser.add_argument('--ratio_train_valid', '-rtv', default=0.8, type=float,
                        help='Ratio between validation and training dataset')
    parser.add_argument('--ratio_Positive_set_to_Unlabeled', '-rpu', default=0.95, type=float,
                         help='Ratio of Positive class that will be set as Negative') 
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Mini batch size')
    parser.add_argument('--Brats2020_is_2d', '-2dBrats', default = True, type= str2bool,
                        help='Determine if a converted 2D Brats2020 must be used, if set to true, convert automatically')
    
    parser.add_argument('--preset', '-p', type=str, default= 'nnPULoss',
                        choices=['nnPULoss','BCELoss', 'FocalLoss'],
                        help="Preset of configuration\n"
                             "nnPULoss: With nnPU criterion\n"
                             "BCELoss: With BinaryCrossEntropy\n"
                             "FocalLoss: With BinaryCrossEntropy\n")
    
    parser.add_argument('--prior', '-pr', default= 0.5, type=float,
                        help='Prior for nnPULoss')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    
    parser.add_argument('--stepsize', '-s', default=1e-4, type=float,
                        help='Stepsize of gradient method')
    
    parser.add_argument('--optimizer','-opti', type=str, default="SGD", choices=['SGD', 'Adam','AdaGrad'],
                        help='Selection of the optimizer')
    
    parser.add_argument('--load_model', '-load_model', default= 'model_saved_Valid/epoch_3.pth', type= str,
                        help='Continue training')
    
    args = parser.parse_args(arguments)
    
    
    args.model_folder_name, args.name = os.path.split(args.load_model)
    args.path_to_last_model = args.load_model
    
    
    # Prior
    if args.prior is not None:
        args.prior = torch.tensor(args.prior, dtype= torch.float)
    

    if args.preset == "nnPULoss":
        args.loss = "nnPULoss"
    elif args.preset == "BCELoss":
        args.loss = "BCELoss"
    elif args.preset =="FocalLoss":
        args.loss = "FocalLoss"
        
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(arguments):
    """ Get the dataloader"""
    args = process_args(arguments)
    train_data, valid_data = select_preprocess(args.Brats2020_is_2d, args.rootdir, args.ratio_train_valid, args.ratio_Positive_set_to_Unlabeled)
    dataloader_data = select_dataloader(args.Brats2020_is_2d, train_data, valid_data, args.preset, 1, True, 8, args.prior)
    _, valid_dataloader, args.prior = dataloader_data
    
    """Get model"""
    model           = unet().to(args.device)
    criterion       = select_loss(args.loss, args.prior, args.beta, args.gamma)
    model.load_state_dict(torch.load(args.path_to_last_model)['model_state_dict'])
    epoch = torch.load(args.path_to_last_model)['epoch']
    
    """Create folder to save the prediction"""        
    folder_name = 'RatioTrainValid {} RatioPosToNeg {} Epoch {}'.format(args.ratio_train_valid, args.ratio_Positive_set_to_Unlabeled, epoch)
    folder_path = os.path.join(args.model_folder_name,'ValidPredOnly', folder_name)
    print(folder_path)
    try:
        os.makedirs(folder_path)
    except:
        pass
  
    loss_array = []
    pred_array = []
    for i, data in enumerate(valid_dataloader):
      input, target = data['img'], data['target']
      input, target = input.to(args.device), target.to(args.device) # Send to device (GPU or CPU)

      with torch.no_grad():
        output        = model(input)

        loss, x_grad  = criterion(output.squeeze(), target.squeeze())
        loss_value    = loss.item()
        # loss_array.append(loss_value)
        # pred_array.append(output.detach().cpu().numpy())
        
        save_name = os.path.join(folder_path, 'prediction_' + str(i)+'.npy')    
        np.save(save_name, output.detach().cpu().numpy())

    
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])