import torch
import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset

class PU_BraTS2020_Dataset_3D(Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target_id       = self.targets[subject_id]
        target_path     = target_id['seg_path']
        p_coordinate    = target_id['P_coordinate']
        unhealthy_slice = target_id['unhealthy_slice']
        
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img_vol = (img_vol/img_vol.max())*255
        img_vol = img_vol.astype(np.uint8)
        
        target_vol = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        
                
        # Only slice that contains unhealhty
        img_slice           = img_vol[slice_id,:,:]
        target_slice_pu     = -1*torch.ones((img_slice.shape[0], img_slice.shape[1]), dtype= torch.float32)
        target_slice        = target_vol[slice_id,:,:]
        target_slice_bin    = np.where(target_slice>0, 1, -1)
        
        # Get Positive pixel coordinate
        p_pixel_index   = np.where(p_coordinate[0] == slice_id)
        if len(p_pixel_index) != 0:               
            for p_index in p_pixel_index:
                y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
                target_slice_pu[y,x]    = 1    
                        

        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        target_slice_pu = torch.tensor(target_slice_pu, dtype= torch.float32)
        target_slice_bin= torch.tensor(target_slice_bin, dtype= torch.float32)

        data = dict({
            'img': img_slice,
            'target': target_slice_pu,
            'original_target': target_slice_bin,   
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data
    

    def get_prior(self) -> float:
        subject = []
        for img_dict in self.inputs:
            subject_id= img_dict['id']
            if subject_id not in subject:
                subject.append(subject_id)

        # Count the number of positive pixel for prior
        nb_positive_pixel = 0
        total_pixel = 0
        for subj in subject:
            target  = self.targets[subj]

            seg                     = target['seg_path']
            unhealthy_slice         = target['unhealthy_slice']
            slice_min, slice_max    = min(unhealthy_slice), max(unhealthy_slice)

            img_vol             = sitk.GetArrayFromImage(sitk.ReadImage(seg))
            img_vol_unhealthy   = img_vol[slice_min:slice_max,:,:]

            nb_p    = np.count_nonzero(img_vol_unhealthy> 0)
            total   = img_vol_unhealthy.size

            nb_positive_pixel   += nb_p
            total_pixel         += total

        prior = torch.tensor(nb_positive_pixel/total_pixel, dtype= torch.float)

        return prior

class PN_BraTS2020_Dataset_3D(Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target_id       = self.targets[subject_id]
        target_path     = target_id['seg_path']
        p_coordinate    = target_id['P_coordinate']
        unhealthy_slice = target_id['unhealthy_slice']
        
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img_vol = (img_vol/img_vol.max())*255
        img_vol = img_vol.astype(np.uint8)
        
        target_vol = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        
                
        # Only slice that contains unhealhty
        img_slice           = img_vol[slice_id,:,:]
        target_slice_pu     = -1*torch.ones((img_slice.shape[0], img_slice.shape[1]), dtype= torch.float32)
        target_slice        = target_vol[slice_id,:,:]
        target_slice_bin    = np.where(target_slice>0, 1, -1)
        
        # Get Positive pixel coordinate
        p_pixel_index   = np.where(p_coordinate[0] == slice_id)
        if len(p_pixel_index) != 0:               
            for p_index in p_pixel_index:
                y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
                target_slice_pu[y,x]    = 1    
                        

        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        target_slice_pu = torch.tensor(target_slice_pu, dtype= torch.float32)
        target_slice_bin= torch.tensor(target_slice_bin, dtype= torch.float32)

        data = dict({
            'img': img_slice,
            'target': target_slice_pu,
            'original_target': target_slice_bin,   
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data

class BCE_BraTS2020_Dataset_3D(Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms        

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target_id       = self.targets[subject_id]
        target_path        = target_id['seg_path']
        unhealthy_slice = target_id['unhealthy_slice']
        
        img_vol     = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img_vol     = (img_vol/img_vol.max())*255
        img_vol     = img_vol.astype(np.uint8)
        img_slice   = img_vol[slice_id,:,:]

        target_vol      = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        target_slice    = target_vol[slice_id,:,:]
        target_slice_bin= np.where(target_slice>0, 1, 0)


        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        target_slice_bin = torch.tensor(target_slice_bin, dtype=torch.float32)   

        data = dict({
            'img': img_slice,
            'target': target_slice_bin,
            'original_target': target_slice_bin,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data
    
    
    # class BraTS2020_Dataset_3D(Dataset):
#     def get_lengths(self) -> None:
#         print('len(data): ', len(self.inputs))
#         print('len(targets): ', len(self.targets))

    # def show_PU_target(self, index: int)-> None:
    #     input       = self.inputs[index]
    #     subject_id  = input['id']
    #     img_path    = input['img_path']
    #     slice_id    = input['slice']
    #     mode        = input['mode']

    #     target          = self.targets[subject_id]
    #     p_coordinate    = target['P_coordinate']
        
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                
    #     # Only slice that contains unhealhty
    #     img_slice   = img[slice_id,:,:]
    #     seg = -1*torch.ones((img_slice.shape[0], img_slice.shape[1]), dtype= torch.float32)

    #     # Get Positive pixel coordinate
    #     p_pixel_index   = np.where(p_coordinate[0] == slice_id)

    #     if len(p_pixel_index) != 0:               
    #         for p_index in p_pixel_index:
    #             y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
    #             seg[y,x]    = 1    

    #     print(subject_id, slice_id)
    #     plt.imshow(seg)

    # def show_PN_target(self, index:int)-> None:
    #     input        = self.inputs[index]
    #     subject_id   = input['id']
    #     slice_id     = input['slice']

    #     target       = self.targets[subject_id]
    #     seg_path     = target['seg_path']
        
    #     seg          = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    #     seg_slice    = seg[slice_id,:,:]

    #     seg_bin = np.where(seg_slice>0, 1, -1)

    #     print(subject_id, slice_id)
    #     plt.imshow(seg_bin)

    # def show_img(self, index: int)-> None:
    #     input       = self.inputs[index]
    #     subject_id  = input['id']
    #     img_path    = input['img_path']
    #     slice_id    = input['slice']
    #     mode        = input['mode']
        
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    #     img_slice = img[slice_id,:,:]

    #     print(subject_id, slice_id, mode)
    #     plt.imshow(img_slice)