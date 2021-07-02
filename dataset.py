import torch
import SimpleITK as sitk

from tqdm import tqdm, trange

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose

class BraTS2020_Dataset(Dataset):
  def get_lengths(self) -> None:
        print('len(data): ', len(self.inputs))
        print('len(targets): ', len(self.targets))

  def show_PU_target(self, index: int)-> None:
      input       = self.inputs[index]
      subject_id  = input['id']
      img_path    = input['img_path']
      slice_id    = input['slice']
      mode        = input['mode']

      target          = self.targets[subject_id]
      p_coordinate    = target['P_coordinate']
      
      img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
              
      # Only slice that contains unhealhty
      img_slice   = img[slice_id,:,:]
      seg = -1*torch.ones((img_slice.shape[0], img_slice.shape[1]), dtype= torch.float32)

      # Get Positive pixel coordinate
      p_pixel_index   = np.where(p_coordinate[0] == slice_id)

      if len(p_pixel_index) != 0:               
          for p_index in p_pixel_index:
              y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
              seg[y,x]    = 1    

      print(subject_id, slice_id)
      plt.imshow(seg)

  def show_PN_target(self, index:int)-> None:
      input        = self.inputs[index]
      subject_id   = input['id']
      slice_id     = input['slice']

      target       = self.targets[subject_id]
      seg_path     = target['seg_path']
      
      seg          = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
      seg_slice    = seg[slice_id,:,:]

      seg_bin = np.where(seg_slice>0, 1, -1)

      print(subject_id, slice_id)
      plt.imshow(seg_bin)


  def show_img(self, index: int)-> None:
      input       = self.inputs[index]
      subject_id  = input['id']
      img_path    = input['img_path']
      slice_id    = input['slice']
      mode        = input['mode']
      
      img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
      img_slice = img[slice_id,:,:]

      print(subject_id, slice_id, mode)
      plt.imshow(img_slice)


class PU_BraTS2020_Dataset(BraTS2020_Dataset, Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target          = self.targets[subject_id]
        p_coordinate    = target['P_coordinate']
        unhealthy_slice = target['unhealthy_slice']
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img/img.max())*255
        img = img.astype(np.uint8)
                
        # Only slice that contains unhealhty
        img_slice   = img[slice_id,:,:]
        seg_slice_bin = -1*torch.ones((img_slice.shape[0], img_slice.shape[1]), dtype= torch.float32)

        # Get Positive pixel coordinate
        p_pixel_index   = np.where(p_coordinate[0] == slice_id)

        if len(p_pixel_index) != 0:               
            for p_index in p_pixel_index:
                y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
                seg_slice_bin[y,x]    = 1    

        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        seg_slice_bin = torch.tensor(seg_slice_bin, dtype= torch.float32)

        data = dict({
            'img': img_slice,
            'target': seg_slice_bin,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data
    

    def get_prior(self) -> float:
        subject = []
        for img_dict in tqdm(self.inputs):
            subject_id= img_dict['id']
            if subject_id not in subject:
                subject.append(id)

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
            total   = img_vol_unhealthy.size()

            nb_positive_pixel   += nb_p
            total_pixel         += total

        prior = torch.tensor(nb_positive_pixel/total_pixel, dtype= torch.float)

        return prior

class PN_BraTS2020_Dataset(BraTS2020_Dataset, Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target          = self.targets[subject_id]
        seg_path        = target['seg_path']
        unhealthy_slice = target['unhealthy_slice']
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img/img.max())*255
        img = img.astype(np.uint8)
        img_slice = img[slice_id,:,:]

        seg           = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        seg_slice     = seg[slice_id,:,:]
        seg_slice_bin = np.where(seg_slice>0, 1, -1)


        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        seg_slice_bin = torch.tensor(seg_slice_bin, dtype=torch.float32)

        data = dict({
            'img': img_slice,
            'target': seg_slice_bin,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data

class BCE_BraTS2020_Dataset(BraTS2020_Dataset, Dataset):
    def __init__(self, data_list, transforms= None) -> None:
        super().__init__()
        self.inputs      = data_list[0]
        self.targets     = data_list[1]
        
        self.transforms  = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple:
        input       = self.inputs[index]
        subject_id  = input['id']
        img_path    = input['img_path']
        slice_id    = input['slice']
        mode        = input['mode']

        target          = self.targets[subject_id]
        seg_path        = target['seg_path']
        unhealthy_slice = target['unhealthy_slice']
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img/img.max())*255
        img = img.astype(np.uint8)
        img_slice = img[slice_id,:,:]

        seg           = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        seg_slice     = seg[slice_id,:,:]
        seg_slice_bin = np.where(seg_slice>0, 1, 0)


        if self.transforms:
            img_slice = self.transforms(img_slice)
        else:
            img_slice = torch.Tensor(img_slice)

        seg_slice_bin = torch.tensor(seg_slice_bin, dtype=torch.float32)   

        data = dict({
            'img': img_slice,
            'target': seg_slice_bin,
            'id': subject_id, 
            'mode': mode,
            'slice': slice_id,
            'unhealthy_slice': unhealthy_slice,
        })
        return data

# def get_PU_dataloader(train_data, valid_data, device):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

#     transforms = Compose([
#     ToTensor(), # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
#     # Normalize(mean=(0.5,), std=(0.5,)),    
#     ])

#     train_dataset   = PU_BraTS2020_Dataset(train_data, transform= transforms),
#     valid_dataset   = PN_BraTS2020_Dataset(valid_data, transform= transforms)

#     prior = train_dataset.get_prior()
            
#     batch_size =   32 #64
#     transforms = Compose([
#     ToTensor(), # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
#     # Normalize(mean=(0.5,), std=(0.5,)),    
#     ])
#     train_dataloader    = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, **kwargs),
#     valid_dataloader    = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False, **kwargs)

#     return train_dataloader, valid_dataloader, prior