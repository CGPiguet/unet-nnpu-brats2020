import os
from numpy.lib.type_check import imag
from pandas.core.algorithms import mode
import torch
import SimpleITK as sitk

from tqdm import tqdm, trange

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

class PU_BraTS2020_Dataset_2D(Dataset):
    def __init__(self, path_to_data, transforms= None) -> None:
        super().__init__()
        self.path_to_data = path_to_data      
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.path_to_data)

    def __getitem__(self, index: int) -> dict:
        data_path = self.path_to_data[index]
        # ['positive_coordinate.npy', 'seg.npy', 'img.npy', 'SubjectInfos.csv']
        image_path          = os.path.join(data_path, 'img.npy')
        ground_truth_path   = os.path.join(data_path, 'seg.npy')
        subjectInfo_path    = os.path.join(data_path, 'SubjectInfos.csv')
        positive_coor_path  = os.path.join(data_path, 'positive_coordinate.npy')

        image               = np.load(image_path)
        target              = np.load(ground_truth_path)
        subjectInfo         = pd.read_csv(subjectInfo_path)
        positive_coor       = np.load(positive_coor_path).squeeze()

        subject_id          = subjectInfo['subject'].unique().item()
        slice_id            = subjectInfo['slice'].unique()
        mode                = subjectInfo['mode'].unique().item()
        unhealthy_slice     = subjectInfo['unhealthy_slice'].unique()

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.Tensor(image)

        target_bin = -1*np.ones((target.shape))
        target_bin[positive_coor[0], positive_coor[1]] = 1
        target_bin = torch.tensor(target_bin, dtype= torch.float32)


        data = dict({
            'img': image,
            'target': target_bin,
            'original_target':target,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data
        

class PN_BraTS2020_Dataset_2D(Dataset):
    def __init__(self, path_to_data, transforms= None) -> None:
        super().__init__()
        self.path_to_data = path_to_data      
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.path_to_data)

    def __getitem__(self, index: int) -> dict:
        data_path = self.path_to_data[index]
        # ['positive_coordinate.npy', 'seg.npy', 'img.npy', 'SubjectInfos.csv']
        image_path          = os.path.join(data_path, 'img.npy')
        ground_truth_path   = os.path.join(data_path, 'seg.npy')
        subjectInfo_path    = os.path.join(data_path, 'SubjectInfos.csv')

        image               = np.load(image_path)
        target              = np.load(ground_truth_path)
        subjectInfo         = pd.read_csv(subjectInfo_path)

        subject_id          = subjectInfo['subject'].unique().item()
        slice_id            = subjectInfo['slice'].unique()
        mode                = subjectInfo['mode'].unique().item()
        unhealthy_slice     = subjectInfo['unhealthy_slice'].unique()

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.Tensor(image)

        target_bin = np.where(target>0, -1, 1)
        target_bin = torch.tensor(target_bin, dtype= torch.float32)


        data = dict({
            'img': image,
            'target': target_bin,
            'original_target':target,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data
        

class BCE_BraTS2020_Dataset_2D(Dataset):
    def __init__(self, path_to_data, transforms= None) -> None:
        super().__init__()
        self.path_to_data = path_to_data      
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.path_to_data)

    def __getitem__(self, index: int) -> dict:
        data_path = self.path_to_data[index]
        # ['positive_coordinate.npy', 'seg.npy', 'img.npy', 'SubjectInfos.csv']
        image_path          = os.path.join(data_path, 'img.npy')
        ground_truth_path   = os.path.join(data_path, 'seg.npy')
        subjectInfo_path    = os.path.join(data_path, 'SubjectInfos.csv')

        image               = np.load(image_path)
        target              = np.load(ground_truth_path)
        subjectInfo         = pd.read_csv(subjectInfo_path)

        subject_id          = subjectInfo['subject'].unique().item()
        slice_id            = subjectInfo['slice'].unique()
        mode                = subjectInfo['mode'].unique().item()
        unhealthy_slice     = subjectInfo['unhealthy_slice'].unique()

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.Tensor(image)

        target_bin = np.where(target>0, 0, 1)
        target_bin = torch.tensor(target_bin, dtype= torch.float32)


        data = dict({
            'img': image,
            'target': target_bin,
            'original_target':target,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data


