import sys
import argparse
import os

import numpy as np
import SimpleITK as sitk

import pandas as pd

from contextlib import redirect_stdout

from preprocess3D import preprocess_brats2020_3D

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Standalone function to convert BraTS2020 to 2D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rootdir', '-r', type=str, default="MICCAI_BraTS2020_TrainingData",
                        help='Root directory of the BraTS2020')
    parser.add_argument('--ratio_train_valid', '-rtv', default=0.8, type=float,
                        help='Ratio between validation and training dataset')
    parser.add_argument('--ratio_Positive_set_to_Unlabeled', '-rpu', default=0.95, type=float,
                         help='Ratio of Positive class that will be set as Negative') 
    args = parser.parse_args(arguments)
    return args
                    

def preprocess_brats2020_2D(root_dir: str = 'MICCAI_BraTS2020_TrainingData', ratio_train_valid: float = 0.8, ratio_P_to_U: float = 0.95)-> tuple(list, list):
    """Verify if a current 2D Brats2020 is currently present in the repertory, if not it converts it automatically
    with ratio of Positive pixel set as Unlabeled and the ratio to divide the dataset between the train and validation dataset. 

    Args:
        root_dir (str): Path to the original BraTS2020 dataset. Defaults to 'MICCAI_BraTS2020_TrainingData'.
        ratio_train_valid (float): Ratio to divide the data between train and validation dataset. Defaults to 0.8.
        ratio_P_to_U (float): Ratio of Positive pixel set as Unlabaled in respect to PU Learning. Defaults to 0.95.

    Returns:
        (list, list): Returns the list of path for the training data and validation data.
    """
    new_rootdir = '2D_BraTS2020 RatioTrainValid ' + str(ratio_train_valid) +' RatioPosToNeg ' + str(ratio_P_to_U)
    if os.path.exists(new_rootdir):
        print("2D BraTS2020 already present with:\n\tRatioTrainValid: {}\tRatioPosToNeg: {}".format(ratio_train_valid, ratio_P_to_U))
    else:
        train_data, valid_data = preprocess_brats2020_3D(root_dir, ratio_train_valid, ratio_P_to_U)
        os.mkdir(new_rootdir)
        convert_BraTS2020_to_2D(new_rootdir, train_data, True, ratio_train_valid, ratio_P_to_U)    
        convert_BraTS2020_to_2D(new_rootdir, valid_data, False, ratio_train_valid, ratio_P_to_U) 
    for train_valid_dir in os.listdir(new_rootdir):
        if train_valid_dir == 'BraTS2020_Train':
            train_data = retrieve_img_path_2D(os.path.join(new_rootdir, train_valid_dir))
        elif train_valid_dir == 'BraTS2020_Valid':
            valid_data = retrieve_img_path_2D(os.path.join(new_rootdir, train_valid_dir))
        else:
            raise NotImplementedError("Not implemented name folder. Should expect BraTS2020_Train or BraTS2020_Valid but got: {}".format(train_valid_dir))

    return train_data, valid_data

def retrieve_img_path_2D(rootdir: os.path)-> list:
    """Retrieve the path to the 2D data folder

    Args:
        rootdir (os.path): Path to the folder that contains the train and valid dataset

    Returns:
        list: list that contains the path to the data
    """
    data = []
    for subjectdir in os.listdir(rootdir):
        subjectdir_path = os.path.join(rootdir,subjectdir)
        if not os.path.isdir(subjectdir_path):
            continue
        for modedir in os.listdir(subjectdir_path):
            modedir_path = os.path.join(subjectdir_path ,modedir)
            if not os.path.isdir(modedir_path):
                continue
            #//TODO Toggle comment to use mode of choice 
            # elif modedir == 'T1':
            #     continue
            elif modedir == 'T1ce':
                continue
            elif modedir == 'T2':
                continue
            elif modedir == 'T2flair':
                continue
            for slicedir in os.listdir(modedir_path):
                slicedir_path = os.path.join(modedir_path, slicedir)
                if not os.path.isdir(slicedir_path):
                    continue
                data.append(slicedir_path)       
    return data

def convert_BraTS2020_to_2D(root_dir: str, data_tuple: tuple, train: bool, ratio_train_valid: float, ratio_P_to_U: float)->None:
    """Create a folder to save the converted 2D BraTS2020 dataset.

    Args:
        root_dir (os.path): Path where the new 2D BraTS dataset will be saved.
        data_tuple (tuple): Contains the data and ground truth of the original 3D Brats2020 dataset.
        train (bool): If set to True, it create the folder for train data. If False, it create the folder for validation data.
        ratio_train_valid (float): Ratio to divide the data between the train and validation dataset.
        ratio_P_to_U (float): Ratio of Positive pixel set to Unlabeled in respect to PU learning.
    """
    

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
        save_path_img = os.path.join(img_slice_folder,'img.jpg')
        save_path_arr = os.path.join(img_slice_folder,'img.npy')
        ### PIL Image
        # img = Image.fromarray(img_slice).convert("L")
        # img.save(save_path_img)
        ### Numpy array
        np.save(save_path_arr, img_slice)

        # SEG
        ## Load segmentation image
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        seg = seg.astype(np.uint8)
        ## Get Positive pixel coordinate
        seg_slice   = seg[slice_id,:,:]
        ## Save SEG 
        save_path_img = os.path.join(img_slice_folder,'seg.jpg')
        save_path_arr = os.path.join(img_slice_folder,'seg.npy')
        ### PIL Image
        # seg = Image.fromarray(seg_slice).convert("L")
        # seg.save(save_path_img)
        ### Numpy array       
        np.save(save_path_arr, seg_slice)


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

        # Save Subject Info 
        save_path = os.path.join(img_slice_folder,'SubjectInfos.csv')
        df = pd.DataFrame({
            'subject': subject_id,
            'slice': slice_id,
            'mode': mode,
            'unhealthy_slice': unhealthy_slice
        })
        df.to_csv(save_path)


def main(arguments):
    args = process_args(arguments)
    preprocess_brats2020_2D(args.rootdir, args.ratio_train_valid, args.ratio_Positive_set_to_Unlabeled)

if __name__ == '__main__':
    main(sys.argv[1:])