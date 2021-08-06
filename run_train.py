import sys
import argparse
import os

import torch
from contextlib import redirect_stdout

# Personal function
from model import unet
from trainer import Trainer
from utils import select_preprocess, select_dataloader, select_loss, print_info_before_training

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
    parser.add_argument('--stepsize', '-s', default=1e-4, type=float,
                        help='Stepsize of gradient method')
    parser.add_argument('--out', '-o', default='/storage/homefs/cp14h011/unet-nnpu-brats2020/model_saved_',
                        help='Directory to output the result')
    parser.add_argument('--validation', '-v', default=False, type= str2bool,
                        help='Use of a validation dataset')
    parser.add_argument('--Brats2020_is_2d', '-2dBrats', default = True, type= str2bool,
                        help='Determine if a converted 2D Brats2020 must be used, if set to true, convert automatically')

    parser.add_argument('--load_checkpoint', '-load_checkpoint', default =None, type= str,
                        help='Continue training by pointing to the correct saved model')
    
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
    if args.load_checkpoint is not None:
        # Retrive the folder name of the loaded model to save in it       
        total_path =  os.path.split(args.out)[1]
        path_to_model_folder =  os.path.split(args.load_checkpoint)[0]
        name = path_to_model_folder.replace(total_path, "")
        args.name = name
        
    args.original_epoch = 0   

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
    optimizer       = torch.optim.SGD(model.parameters(), lr = args.stepsize,  weight_decay=0.005)
    criterion       = select_loss(args.loss, args.prior, args.beta, args.gamma)
    original_epoch  = 0
    
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint['model_state_dict']))
        optimizer.load_state_dict(torch.load(args.load_checkpoint['optimizer_state_dict']))
        args.original_epoch = torch.load(args.load_checkpoint['epoch'])
        
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