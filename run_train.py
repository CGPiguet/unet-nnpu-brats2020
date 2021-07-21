import sys
import argparse
import os



from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

import pandas as pd

# Personal function
from preprocess import preprocess_brats2020
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
    # parser.add_argument('--labeled', '-l', default=100, type=int,
    #                     help='# of labeled data')
    # parser.add_argument('--unlabeled', '-u', default=59900, type=int,
    #                      help='# of unlabeled data')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
    parser.add_argument('--loss', type=str, default="nnPULoss", choices=['nnPULoss', 'BCELoss','FocalLoss'],
                        help='The name of a loss function')
    # parser.add_argument('--nnPUloss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
    #                     help='The name of a loss function used in nnPU')
    # parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
    #                     help='The name of a classification model')
    parser.add_argument('--stepsize', '-s', default=1e-4, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='/results/',
                        help='Directory to output the result')
    parser.add_argument('--validation', '-v', default=False, type= str2bool,
                        help='Use of a validation dataset')

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

    # Prior
    if args.prior is not None:
        args.prior = torch.tensor(args.prior, dtype= torch.float)

    assert (args.batchsize > 0)
    assert (args.epoch > 0)
    assert (0. <= args.beta)
    # assert (0. <= args.gamma <= 1.)
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


def print_info_before_training(args):
    print("")
    print("Name of the Job/training: {}".format(args.name))
    print("Device: {}".format(args.device))
    print("Preset: {}".format(args.preset))
    print("prior: {}".format(args.prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("lr: {}".format(args.stepsize))
    print("beta: {}".format(args.beta))
    print("validation dataset: {}".format(args.validation))
    print("Num of Workers: {}".format(args.num_worker))
    print("")


def run_trainer(arguments):
    print("\nTrainer setup\n")
    args = process_args(arguments)

    train_data, valid_data = preprocess_brats2020(root_dir=args.rootdir, ratio_train_valid= 0.8, ratio_P_to_U= 0.95)

    dataloader_data = select_dataloader(train_data,valid_data, args.preset, args.batchsize, args.validation, args.num_worker, args.prior)
    train_dataloader, valid_dataloader, args.prior = dataloader_data

    model       = unet().to(args.device)
    optimizer   = torch.optim.SGD(model.parameters(), lr = args.stepsize,  weight_decay=0.005)
    criterion   = select_loss(args.loss, args.prior, args.beta, args.gamma)
    
    kwargs =  {
        'criterion': criterion,
        'optimizer': optimizer,
        'train_Dataloader': train_dataloader,
        'valid_Dataloader': valid_dataloader,
        'epochs': args.epoch
        }
    print_info_before_training(args)
    trainer = Trainer(args.name, model, args.device, **kwargs)

    trainer.run_trainer()

    
    if args.validation is not None:
        df = pd.DataFrame({
            'train_loss': trainer.train_loss,
            'train_dice': trainer.train_dice_coef,
            'valid_loss': trainer.valid_loss,
            'valid_dice': trainer.valid_dice_coef
        })
    else :
        df = pd.DataFrame({
            'train_loss': trainer.train_loss,
            'train_dice': trainer.train_dice_coef,
        })
    folder_name = '/storage/homefs/cp14h011/unet-nnpu-brats2020/resultsCSV/'
    file_name   = trainer.name + '.csv'
    save_name   = os.path.join(folder_name, file_name)

    df.to_csv(save_name)


if __name__ == '__main__':
    run_trainer(sys.argv[1:])