import torch 
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

from typing import Tuple
# Personal function
from preprocess3D import preprocess_brats2020_3D
from preprocess2D import preprocess_brats2020_2D
from dataset3D import BCE_BraTS2020_Dataset_3D, PN_BraTS2020_Dataset_3D, PU_BraTS2020_Dataset_3D
from dataset2D import BCE_BraTS2020_Dataset_2D, PN_BraTS2020_Dataset_2D, PU_BraTS2020_Dataset_2D

from nnPULoss import PULoss
from FocalLoss import BinaryFocalLossWithLogits

def select_optimizer(optimizer_name: str, learning_rate: float, model: nn.Module):
    """Select and return the chosen optimizer.

    Args:
        optimizer_name (str): The optimizer name that help to select the optimizer in respect to the name.
        learning_rate (float): Learning rate 
        model (nn.Module): Model

    Raises:
        NotImplementedError: Raise an error if the optimizer name is not recognized.

    """
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate,  weight_decay=0.005)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,  weight_decay=0.005)
    elif optimizer_name == 'AdaGrad':
        optimizer = torch.optim.AdaGrad(model.parameters(), lr = learning_rate,  weight_decay=0.005)
    else:
        raise NotImplementedError('No optimizer found with the name: {}. Please select either SGD, Adam, AdaGrad'.format(optimizer_name))
    return optimizer
        
def select_loss(loss_name: str, prior: float, beta: float, gamma: float)-> torch.nn.Module:
    """Simply select the loss between BCELossWithLogitsLoss, FocalLoss (deprecated) and nnPU Loss.

    Args:
        loss_name (str): The name of the loss to use
        prior (float): The Positive prior, if not given, it is compute from the dataset to
        beta (float): Beta parameter of nnPU
        gamma (float): Gamma parameter of NnPU

    Returns:
        torch.nn Module: Loss function 
    """    
    if loss_name == "nnPULoss":
        loss_fn = PULoss(prior, beta= beta, gamma= gamma)
    elif loss_name == "BCELoss":
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_name == "FocalLoss":
        loss_fn = BinaryFocalLossWithLogits(reduction='mean')
    return loss_fn

def select_preprocess(img_mode: str, Brats2020_is_2d: bool , root_dir: str, ratio_train_valid: float, ratio_Positive_set_to_Unlabeled: float) -> Tuple[list, list]:
    """Select the preprocess, between the 3D BraTS2020 dataset or the 2D converted BraTS2020.

    Args:
        Brats2020_is_2d (bool): If set to True, use the converted 2D BraTS2020 dataset. If set to False, use the original 3D BraTS2020 dataset.
        root_dir (os.path): Directory of the BraTS2020 dataset
        ratio_train_valid (float): Ratio to divide the BraTS2020 dataset between train and validation dataset.
        ratio_Positive_set_to_Unlabeled (float): Ratio of Positive pixel set as Unlabeled in respect to PU learning.

    Returns:
        Tuple[list, list]: Returns two list for training and validation data
    """    
    if Brats2020_is_2d:
        train_data, valid_data = preprocess_brats2020_2D(img_mode, root_dir, ratio_train_valid, ratio_Positive_set_to_Unlabeled)

    else:
        train_data, valid_data = preprocess_brats2020_3D(img_mode, root_dir, ratio_train_valid, ratio_Positive_set_to_Unlabeled)
    return train_data, valid_data



def select_dataloader(Brats2020_is_2d: bool,
                      train_data: list,
                      valid_data: list,
                      dataloader_preset: str,
                      batchsize: int, is_validation: bool,
                      num_worker: int, prior: float)-> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float]:
    """Select the correct dataloader in respect to the given parameters.

    Args:
        Brats2020_is_2d (bool): If set as True, use the 2D converted BraTS2020 dataset. If set as False, use the the 3D original BraTS2020 dataset.
        train_data (list): List of the training data path.
        valid_data (list): List of the validation data path.
        dataloader_preset (str): Define which dataloader to use in respect to the Loss function used.
        batchsize (int): Size of the batch  
        is_validation (bool): If set to True, use a validation dataset. If set to False, only a training dataset will be used.
        num_worker (int): Set the num_worker parameter of the Dataloader.
        prior (float): Positive-class prior parameter from nnPU.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float]: Returns the two Dataloaders and the prior.
    """
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

    if Brats2020_is_2d:
        if dataloader_preset == "nnPULoss":
            train_dataset       = PU_BraTS2020_Dataset_2D(train_data, transforms= transforms)  
            train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)     
            if is_validation:
                valid_dataset       = PN_BraTS2020_Dataset_2D(valid_data, transforms= transforms)
                valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
            else:
                valid_dataloader = None
        elif dataloader_preset == "BCELoss" or "FocalLoss":
            train_dataset       = BCE_BraTS2020_Dataset_2D(train_data, transforms= transforms)
            train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)       
            if is_validation:
                valid_dataset       = BCE_BraTS2020_Dataset_2D(valid_data, transforms= transforms)
                valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
            else:
                valid_dataloader = None
        else:
            raise ValueError('Unidentified preset has been chosen ')
    else:
        if dataloader_preset == "nnPULoss":
            train_dataset       = PU_BraTS2020_Dataset_3D(train_data, transforms= transforms)  
            train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)     
            if is_validation:
                valid_dataset       = PN_BraTS2020_Dataset_3D(valid_data, transforms= transforms)
                valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
            else:
                valid_dataloader = None
        elif dataloader_preset == "BCELoss" or "FocalLoss":
            train_dataset       = BCE_BraTS2020_Dataset_3D(train_data, transforms= transforms)
            train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs)       
            if is_validation:
                valid_dataset       = BCE_BraTS2020_Dataset_2D(valid_data, transforms= transforms)
                valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)
            else:
                valid_dataloader = None
        else:
            raise ValueError('Unidentified preset has been chosen ')


    if prior is None:
        print('Positive Prior automatically computed')
        prior = train_dataset.get_prior() if dataloader_preset == "nnPULoss" else None

    
    print('IMG status:', train_dataset[0]['img'].shape, train_dataset[0]['img'].dtype, train_dataset[0]['img'].type())
    print('SEG status:', train_dataset[0]['target'].shape, train_dataset[0]['target'].dtype, train_dataset[0]['target'].type())


    return train_dataloader, valid_dataloader, prior


def print_info_before_training(args):
    print("")
    print("Name of the Job/training: {}".format(args.name))
    print("Device:\t {}".format(args.device))
    print("Preset:\t {}".format(args.preset))
    print("loss:\t {}".format(args.loss))
    print("Optimizer:\t {}".format(args.optimizer))
    print("Image modality:\t {}".format(args.image_modality))
    
    print("")
    print("prior:\t {}".format(args.prior))
    print("beta from nnPULoss:\t {}".format(args.beta))
    print("gamma from nnPULoss:\t {}".format(args.gamma))
    print("")
    print("Epoch:\t {}".format(args.epoch))
    print("batchsize:\t {}".format(args.batchsize))
    print("lr:\t {}".format(args.stepsize))
    print("Num of Workers:\t {}".format(args.num_worker))
    print("")
    print("validation dataset:\t {}".format(args.validation))
    print("Use a converted 2D BraTS2020 dataset: {}".format(args.Brats2020_is_2d))
    print("")
    print("Ratio to seperate data into train and validation dataset:\t {}".format(args.ratio_train_valid))
    print("Ratio of Positive voxel set as Neative:\t {}".format(args.ratio_Positive_set_to_Unlabeled))
    print("")
    print("Model load:\t {}".format(args.path_to_last_model))
    print("# of epoch of the loaded model:\t {}".format(args.original_epoch))
    print("")
