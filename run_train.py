import sys
import argparse
import os



from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

import pandas as pd

from contextlib import redirect_stdout

# Personal function
from preprocess3D import preprocess_brats2020_3D
from preprocess2D import preprocess_brats2020_2D

from dataset import PN_BraTS2020_Dataset, PU_BraTS2020_Dataset, BCE_BraTS2020_Dataset
from model import unet
from nnPULoss import PULoss
from trainer import Trainer
from FocalLoss import BinaryFocalLossWithLogits

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rootdir', '-r', type=str, default="MICCAI_BraTS2020_TrainingData",
                        help='Root directory of the BraTS2020')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Name of the job/training.')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Mini batch size')
    parser.add_argument('--device', '-d', type=torch.device, default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Determine the torch device, AutoDetection per default')
    parser.add_argument('--preset', '-p', type=str, default= 'nnPULoss',
                        choices=['nnPULoss','BCELoss', 'FocalLoss'],
                        help="Preset of configuration\n"
                             "nnPULoss: With nnPU criterion\n"
                             "BCELoss: With BinaryCrossEntropy\n"
                             "FocalLoss: With BinaryCrossEntropy\n")
    parser.add_argument('--num_worker', '-nw', default=8, type=int,
                        help='Number of worker in Dataloader')

    parser.add_argument('--prior', '-pr', default=None, type=float,
                        help='Prior for nnPULoss')
                                           
    parser.add_argument('--ratio_train_valid', '-rtv', default=0.8, type=float,
                        help='Ratio between validation and training dataset')
    parser.add_argument('--ratio_Positive_set_to_Unlabeled', '-rpu', default=0.95, type=float,
                         help='Ratio of Positive class that will be set as Negative')

    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    # parser.add_argument('--loss', type=str, default="nnPULoss", choices=['nnPULoss', 'BCELoss','FocalLoss'],
    #                     help='The name of a loss function')
    # parser.add_argument('--nnPUloss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
    #                     help='The name of a loss function used in nnPU')
    # parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
    #                     help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-4, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='/storage/homefs/cp14h011/unet-nnpu-brats2020/model_saved_',
                        help='Directory to output the result')
    parser.add_argument('--validation', '-v', default=False, type= str2bool,
                        help='Use of a validation dataset')
    parser.add_argument('--Brats2020_2d', '-2dBrats', default =False, type= str2bool,
                        help='Determine if the converted 2D Brats2020 must be used')

    args = parser.parse_args(arguments)
    # Preset 
    if args.name == None:
        args.name = args.preset
    if args.preset == "nnPULoss":
        args.loss = "nnPULoss"
    elif args.preset == "BCELoss":
        args.loss = "BCELoss"
    elif args.preset =="FocalLoss":
        args.loss = "FocalLoss"

    # if args.Brats2020_2d == True:
    #     temp_name = '2D_BraTS2020 RatioTrainValid ' + str(args.ratio_train_valid) +' RatioPosToNeg ' + str(args.ratio_Positive_set_to_Unlabeled)
    #     args.rootdir = temp_name
    
    # Prior
    if args.prior is not None:
        args.prior = torch.tensor(args.prior, dtype= torch.float)

    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
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


def select_loss(loss_name, prior, beta, gamma):
    if loss_name == "nnPULoss":
        loss_fn = PULoss(prior, beta= beta, gamma= gamma)
    elif loss_name == "BCELoss":
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_name == "FocalLoss":
        loss_fn = BinaryFocalLossWithLogits(reduction='mean')
    return loss_fn

def select_preprocess(Brats2020_2d, root_dir, ratio_train_valid, ratio_Positive_set_to_Unlabeled):
    if Brats2020_2d:
        train_data, valid_data = preprocess_brats2020_3D(root_dir, ratio_train_valid, ratio_Positive_set_to_Unlabeled)

    else:
        train_data, valid_data = preprocess_brats2020_3D(root_dir, ratio_train_valid, ratio_Positive_set_to_Unlabeled)



def select_dataloader(train_data, valid_data, dataloader_preset, batchsize, is_validation, num_worker, prior):
    kwargs = {'num_workers': num_worker, 'pin_memory': True} if torch.cuda.is_available() else {}

    transforms = Compose([
    ToTensor(), # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
    # Normalize(mean=(0.5,), std=(0.5,)),    
    ])
       
    batch_size =   batchsize
    transforms = Compose([
    ToTensor(), # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
    # Normalize(mean=(0.5,), std=(0.5,)),    
    ])

    if dataloader_preset == "nnPULoss":
        train_dataset       = PU_BraTS2020_Dataset(train_data, transforms= transforms)  
        train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)     
        if is_validation:
            valid_dataset       = PN_BraTS2020_Dataset(valid_data, transforms= transforms)
            valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
        else:
            valid_dataloader = None
    elif dataloader_preset == "BCELoss" or "FocalLoss":
        train_dataset       = BCE_BraTS2020_Dataset(train_data, transforms= transforms)
        train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)       
        if is_validation:
            valid_dataset       = BCE_BraTS2020_Dataset(valid_data, transforms= transforms)
            valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
        else:
            valid_dataloader = None
    else:
        raise ValueError('Unidentified preset has been chosen ')

    if prior is None:
        print('Positive Prior automatically computed')
        prior = train_dataset.get_prior() #if dataloader_preset == "nnPU" else None

    
    print('IMG status:', train_dataset[0]['img'].shape, train_dataset[0]['img'].dtype, train_dataset[0]['img'].type())
    print('SEG status:', train_dataset[0]['target'].shape, train_dataset[0]['target'].dtype, train_dataset[0]['target'].type())


    return train_dataloader, valid_dataloader, prior


def save_results(trainer, is_validation, args_out, args_name):
    if is_validation:
        print(len(trainer.train_loss), len(trainer.train_dice_coef), len(trainer.valid_loss),len(trainer.valid_dice_coef))
        df = pd.DataFrame({
            'train_loss': trainer.train_loss,
            'train_dice': trainer.train_dice_coef,
            'train_ROC_AUC': trainer.train_ROC,
            'valid_loss': trainer.valid_loss,
            'valid_dice': trainer.valid_dice_coef,
            'valid_ROC_AUC': trainer.valid_ROC
        })
    else :
        print(len(trainer.train_loss), len(trainer.train_dice_coef))
        df = pd.DataFrame({
            'train_loss': trainer.train_loss,
            'train_dice': trainer.train_dice_coef,
            'train_ROC_AUC': trainer.train_ROC
        })

    folder_name = args_out + args_name
    file_name   = 'results.csv'
    save_name   = os.path.join(folder_name, file_name)

    df.to_csv(save_name)


def print_info_before_training(args):
    print("")
    print("Name of the Job/training: {}".format(args.name))
    print("Device:\t {}".format(args.device))
    print("Preset:\t {}".format(args.preset))
    print("prior:\t {}".format(args.prior))
    print("loss:\t {}".format(args.loss))
    print("Epoch:\t {}".format(args.epoch))
    print("batchsize:\t {}".format(args.batchsize))
    print("lr:\t {}".format(args.stepsize))
    print("beta from nnPULoss:\t {}".format(args.beta))
    print("gamma from nnPULoss:\t {}".format(args.gamma))
    print("validation dataset:\t {}".format(args.validation))
    print("Num of Workers:\t {}".format(args.num_worker))
    print("Ratio to seperate data into train and validation dataset:\t {}".format(args.ratio_train_valid))
    print("Ratio of Positive voxel set as Neative:\t {}".format(args.ratio_Positive_set_to_Unlabeled))
    print("")


def run_trainer(arguments):
    print("\nTrainer setup\n")
    args = process_args(arguments)

    train_data, valid_data = select_preprocess(args.rootdir, args.ratio_train_valid, args.ratio_Positive_set_to_Unlabeled)

    dataloader_data = select_dataloader(train_data,valid_data, args.preset, args.batchsize, args.validation, args.num_worker, args.prior)
    train_dataloader, valid_dataloader, args.prior = dataloader_data

    model       = unet().to(args.device)
    print("model.is_cuda: {}".format(next(model.parameters()).is_cuda))
    optimizer   = torch.optim.SGD(model.parameters(), lr = args.stepsize,  weight_decay=0.005)
    criterion   = select_loss(args.loss, args.prior, args.beta, args.gamma)
    
    kwargs =  {
        'name': args.name,
        'model': model, 
        'criterion': criterion,
        'optimizer': optimizer,
        'train_Dataloader': train_dataloader,
        'valid_Dataloader': valid_dataloader,
        'epochs': args.epoch,
        'device': args.device,
        'out': args.out,
        }

    """Create a folder to save model and parameters"""
    folder_name = args.out + args.name
    file_name   = 'parameter.txt'
    try:
        os.mkdir(folder_name)
    except:
        pass

    """Print info and save info"""
    print_info_before_training(args)

    with open(os.path.join(folder_name,file_name), 'w') as f:
        with redirect_stdout(f):
            print_info_before_training(args)



    trainer = Trainer(**kwargs)
    trainer.run_trainer()

    """Save results"""
    save_results(trainer, args.validation, args.out, args.name)

if __name__ == '__main__':
    run_trainer(sys.argv[1:])