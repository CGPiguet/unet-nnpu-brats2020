import sys
import argparse
import os

import torch
from contextlib import redirect_stdout

# Personal function
from model import unet
from trainer import Trainer
from utils import select_optimizer, select_preprocess, select_dataloader, select_loss, print_info_before_training

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rootdir', '-r', type=str, default="MICCAI_BraTS2020_TrainingData",
                        help='Root directory of the BraTS2020')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Name of the job/training.')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
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
    parser.add_argument('--optimizer','-opti', type=str, default="SGD", choices=['SGD', 'Adam','AdaGrad'],
                        help='Selection of the optimizer')
    parser.add_argument('--stepsize', '-s', default=1e-4, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='model_saved_',
                        help='Directory to output the result')
    parser.add_argument('--validation', '-v', default=False, type= str2bool,
                        help='Use of a validation dataset')
    parser.add_argument('--Brats2020_is_2d', '-2dBrats', default = True, type= str2bool,
                        help='Determine if a converted 2D Brats2020 must be used, if set to true, convert automatically')

    parser.add_argument('--continue_training', '-continue_training', default = False, type= str2bool,
                        help='Continue training')
    
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
        
    # Load previous model 
    args.path_to_last_model = None
    args.original_epoch = 0  
    if args.continue_training == True:
        folder_name = args.out + args.name
        if os.path.isdir(folder_name):
            saved_model= []
            saved_model_name = [files for files in os.listdir(folder_name) if '.pth' in files]
            if not len(saved_model_name):
                raise NotImplementedError('No previous model found to load')
            for models in saved_model_name:
                epoch = int(''.join(char for char in models if char.isdigit()))
                saved_model.append((epoch, models))
            saved_model.sort()
            last_model = saved_model[-1][1]
            path_to_last_model = os.path.join(folder_name, last_model)
        else:
            raise NotImplementedError("No correct folder detected to continue training. Got the name '{}' and didnt' find a folder named '{}'".format(args.name, folder_name))
        args.path_to_last_model = path_to_last_model 
       

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


def run_trainer(arguments):
    print("\nTrainer setup\n")
    args = process_args(arguments)

    train_data, valid_data = select_preprocess(args.Brats2020_is_2d, args.rootdir, args.ratio_train_valid, args.ratio_Positive_set_to_Unlabeled)

    dataloader_data = select_dataloader(args.Brats2020_is_2d, train_data,valid_data, args.preset, args.batchsize, args.validation, args.num_worker, args.prior)
    train_dataloader, valid_dataloader, args.prior = dataloader_data

    """Setup of the model and optimizer parameters"""
    model           = unet().to(args.device)
    optimizer       = select_optimizer(args.optimizer, args.stepsize, model)
    criterion       = select_loss(args.loss, args.prior, args.beta, args.gamma)
    
    if args.continue_training:
        model.load_state_dict(torch.load(args.path_to_last_model)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.path_to_last_model)['optimizer_state_dict'])
        args.original_epoch = torch.load(args.path_to_last_model)['epoch']
        
    print("model.is_cuda: {}".format(next(model.parameters()).is_cuda))
    
    
    kwargs =  {
        'name': args.name,
        'model': model, 
        'criterion': criterion,
        'optimizer': optimizer,
        'train_Dataloader': train_dataloader,
        'valid_Dataloader': valid_dataloader,
        'original_epoch': args.original_epoch,
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
    # save_results(trainer, args.validation, args.out, args.name)

if __name__ == '__main__':
    run_trainer(sys.argv[1:])