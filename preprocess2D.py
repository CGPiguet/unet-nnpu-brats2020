import os
import numpy as np
import SimpleITK as sitk

from PIL import Image
from contextlib import redirect_stdout

from preprocess3D import preprocess_brats2020_3D


def preprocess_brats2020_2D(root_dir: str = 'MICCAI_BraTS2020_TrainingData', ratio_train_valid: float = 0.8, ratio_P_to_U: float = 0.95):
    train_data, valid_data = preprocess_brats2020_3D(root_dir, ratio_train_valid, ratio_P_to_U)
    new_rootdir = '2D_BraTS2020 RatioTrainValid ' + str(ratio_train_valid) +' RatioPosToNeg ' + str(ratio_P_to_U)
    os.mkdir(new_rootdir)
    convert_BraTS2020_to_2D(new_rootdir, train_data, True, ratio_train_valid, ratio_P_to_U)    
    convert_BraTS2020_to_2D(new_rootdir, valid_data, False, ratio_train_valid, ratio_P_to_U) 

    return train_data, valid_data


def convert_BraTS2020_to_2D(root_dir: str, data_tuple: tuple, train: bool, ratio_train_valid: float, ratio_P_to_U: float):

    data, target = data_tuple

    if train:
        main_folder = os.path.join(root_dir,'BraTS2020_Train')
        os.mkdir(main_folder)
    else:
        main_folder = os.path.join(root_dir, 'BraTS2020_Valid')
        os.mkdir(main_folder)

    with open(os.path.join(main_folder,'_datasetParam.txt'), 'w') as f:
        with redirect_stdout(f):
            print('Ratio to separate train and valid data: {}'.format(ratio_train_valid))
            print('Ratio of Positive Voxel set as Negative: {}'.format(ratio_P_to_U))

    

    for img_info in data:
        # Retrieval of all information
        subject_id  = img_info['id']
        img_path    = img_info['img_path']
        slice_id    = img_info['slice']
        mode        = img_info['mode']

        target_temp     = target[subject_id]
        seg_path        = target_temp['seg_path']
        p_coordinate    = target_temp['P_coordinate']
        unhealthy_slice = target_temp['unhealthy_slice']

        # Make directory per subject
        subject_folder = os.path.join(main_folder, str(subject_id))
        try:
            os.mkdir(subject_folder)
        except:
            pass
        
        # Make directory per img mode
        img_mode_folder = os.path.join(subject_folder, str(mode))
        try:
            os.mkdir(img_mode_folder)
        except:
            pass

        # Make dirctory per slice
        img_slice_folder = os.path.join(img_mode_folder, str(slice_id))
        try:
            os.mkdir(img_slice_folder)
        except:
            pass
     
        # IMG
        ## Normalize img 
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img/img.max())*255
        img = img.astype(np.uint8)                
        ## Only slice that contains unhealhty
        img_slice   = img[slice_id,:,:]
        ## Save IMG
        img = Image.fromarray(img_slice).convert("L")
        save_path_img = os.path.join(img_slice_folder,'img.jpg')
        save_path_arr = os.path.join(img_slice_folder,'img.npy')
        img.save(save_path_img)
        np.save(save_path_arr, img_slice)

        # SEG
        ## Load segmentation image
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        ## Get Positive pixel coordinate
        seg_slice   = seg[slice_id,:,:]
        ## Save SEG 
        seg = Image.fromarray(seg_slice).convert("L")
        save_path_img = os.path.join(img_slice_folder,'seg.jpg')
        save_path_arr = os.path.join(img_slice_folder,'seg.npy')
        seg.save(save_path)
        np.save(save_path_arr, img_slice)
        # cv2.imwrite(os.path.join(img_slice_folder,'seg.jpg'), seg_slice)

        # Save Positive Coordinate
        ## Get Positive pixel coordinate
        positive_coordinate = []
        p_pixel_index   = np.where(p_coordinate[0] == slice_id)
        if len(p_pixel_index) != 0:               
            for p_index in p_pixel_index:
                y,x         = p_coordinate[1][p_index], p_coordinate[2][p_index]
                positive_coordinate.append((y,x))
        save_path = os.path.join(img_slice_folder,'positive_coordinate.npy')
        np.save(save_path, positive_coordinate)