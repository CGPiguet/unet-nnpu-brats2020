import os
import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
from tqdm.auto import tqdm, trange
import SimpleITK as sitk


def get_subfolder(root_dir: str):
    """Returns a list with sub-directory of root_dir 

    Args:
        root_dir (str): path to the directory

    Returns:
        list: list of sub-directory of root_dir
    """
    folder_path = []
    for folder_name in os.listdir(root_dir):
        # Iterate through each folders/classes
        if not os.path.isdir(os.path.join(root_dir, folder_name)):
            #Verifiy that the dir root_dir/folder_name exsit
            continue
        path = os.path.join(root_dir,folder_name)
        folder_path.append(path)
    return folder_path



def split_data(list_dir: list, ratio: float = 0.8):
    """Split the data to create a training and validation set

    Args:
        list_dir (list): list of PATHs to the data
        ratio (float): Ratio between training and validation

    Returns:
        tuple: list of PATH for training data and validation data
    """

    n_train = round(len(list_dir)*ratio)
    list_train = []
    list_valid = []

    for i, folder_path  in enumerate(list_dir):
        if i >= n_train:
            list_valid.append(folder_path)
        else:
            list_train.append(folder_path)

    to_print= '\t\tTotal:\t{}\n\t\tRatio:\t{}\n\t\t#train:\t{}\n\t\t#valid:\t{} '
    print(to_print.format(len(list_dir),ratio, len(list_train),len(list_valid)))

    return list_train, list_valid



def get_img_path_from_folder(path_list: list):
    """Return img path from each subject/sub-directory

    Args:
        path_list (list): list of path to each subject/sub-directory

    Returns:
        list: list of dict containing the id of the patient and all its img
    """

    data_list = []
    for folder_path in path_list:

        file_dict = {}
        subject_id = os.path.basename(os.path.normpath(folder_path))
        file_dict.update({'id': subject_id})

        for file_name in os.listdir(folder_path):
            if 't1.nii.gz' in file_name:
                file_dict.update({'T1': os.path.join(folder_path,file_name)})
            # if 't1ce.nii.gz' in file_name:
            #   file_dict.update({'T1ce': os.path.join(folder_path,file_name)})
            # if 't2.nii.gz' in file_name:
            #   file_dict.update({'T2': os.path.join(folder_path,file_name)})
            # if 'flair.nii.gz' in file_name:
            #   file_dict.update({'T2flair': os.path.join(folder_path,file_name)})
            if 'seg.nii.gz' in file_name:
                file_dict.update({'Seg': os.path.join(folder_path,file_name)})

        data_list.append(file_dict)
    return data_list



def get_unhealthy_slice(data: list):
    """Determine unhealthy slice in volumetric data

    Args:
        data (list): list of data containing dict with multiple img and the ground truth

    Returns:
        dict: the key is the id of the subject and the values are a tuple of min/max slice
    """

    unhealthy_dict  = {}

    # progressbar = tqdm(data, desc='Get Unhealhty Slice',total= len(data), leave= True, position=0)
    for subject_dict in data:
        # Loop through all subject 
        unhealthy_slice = []
        target_path     = subject_dict['Seg']
        subject_id      = subject_dict['id']

        target_img  = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        
        
        img_depth = target_img.shape[0] # Get the depth 


        # Save each slice that has unhealthy 
        for slce in range(img_depth):
          target_slice = target_img[slce,:,:]
          if not np.all(target_slice == 0):
              unhealthy_slice.append(slce)

        min, max = np.min(unhealthy_slice), np.max(unhealthy_slice)
        unhealthy_dict.update(dict({
            subject_id: np.array((min, max))
        }))
        # progressbar.update()

    return unhealthy_dict



def Slice3Dto2D(data_list: list, unhealthy_slice_list: list, ratio_P_to_U: float= 0.95):
    """Preprocessing of the 3D image for its future 
        conversion to 2D. It generate the img list that contains a dict:
        - The id of the subject
        - The img_path 
        - The corresponding slice in 2D
        - Image mode
        And it save a target_dict with the ID of the subjects as key and a tuple as values:
        - The img_path to the ground_truth
        - The coordinate of the Positive P data in prevision of PU Learning.


    Args:
        data_list (list): list of dict that containts a dict of all img mode of the subject.
        unhealthy_slice_list (list): Contains the slice of interest. the ones that contains tumor tissue.
        ratio_P_to_U (float): Ratio of P data that must be set to U

    Returns:
        tuple: (input and target)
    """
    input   = []
    target  = {}

    # progressbar = tqdm(data_list, 'Subject', total=len(data_list), leave= True, position=0)
    for data in data_list:      
        data_copy       = data.copy()
        subject_id      = data_copy.pop('id')
        # progressbar.set_description(f'{subject_id}')

        segmentation    = data_copy.pop('Seg')

        unhealthy_slice = unhealthy_slice_list[subject_id]
        slice_max       = max(unhealthy_slice)
        slice_min       = min(unhealthy_slice)

        target_vol      = sitk.GetArrayFromImage(sitk.ReadImage(segmentation))
    
        positive_data_coordinate = set_negative_data(target_vol, ratio_P_to_U)

        for key, img_mode_path in data_copy.items():
            for img_slice in range(slice_min,slice_max+1):
                img_dict = dict({
                    'id': subject_id,
                    'img_path': img_mode_path,
                    'slice': img_slice,
                    'mode': key,
                })
                input.append(img_dict)

        target.update(dict({
        subject_id: dict({
            'seg_path': segmentation,
            'P_coordinate': positive_data_coordinate,
            'unhealthy_slice': unhealthy_slice
            })
        }))
        # progressbar.update()

    return (input, target)

        

def set_negative_data(vol_img: np.ndarray, percentage: float= 0.95):
    """Binarize the ground-truth and set a certain amount of P data (+1) to U (-1) in prevision of PU learning.

    Args:
        vol_img (np.ndarray): Volumetric imag of the ground-truth
        percentage (float, optional): Ratio of P data that need to be set as U. Defaults to 0.95.

    Returns:
        Tuple: Tuple that contains the coordinate of the P data. 
    """
    unhealthy_img   = vol_img

    bin_img         = np.where(unhealthy_img > 0, 1, -1)

    total_positive  = np.count_nonzero(bin_img>0)
    # print('Original number of positive :',np.count_nonzero(bin_img>0) )


    # Shuffle the iterator 
    iterator         = np.argwhere(bin_img>0)
    np.random.shuffle(iterator)

    counter = 0
    while np.floor(total_positive*percentage)> counter:
        i, j, k = iterator[counter]
        bin_img[i,j,k]  = -1
        counter += 1

    positive_coordinate = np.nonzero(bin_img>0)

    # print('After set to negative,number of positive :',np.count_nonzero(bin_img>0), len(positive_coordinate[0]) )

    return positive_coordinate 


def preprocess_brats2020(root_dir: str, ratio_train_valid: float = 0.8, ratio_P_to_U: float = 0.95):
    """General Function to get and preprocess the brats2020 dataset for PU learning

    Args:
        root_dir (str): Root to the BraTS2020 
        ratio_train_valid (float, optional): Ratio to create train and valid dataset. Defaults to 0.8.
        ratio_P_to_U (float, optional): Ratio of P to be set as U. Defaults to 0.95.

    Returns:
        train_data, valid_data: train and validation dataset.
    """

    print("\nPreprocessing")
    print('\tStep 1.\tGet subject folder path from the root')
    folder_path = get_subfolder(root_dir)

    print('\tStep 2.\tSplit data for train and valid dataset')
    train_subjects, valid_subjects = split_data(folder_path, ratio_train_valid)
    
    print('\tStep 3.\tGet img mode from all subject/folder')
    train_list = get_img_path_from_folder(train_subjects)
    valid_list = get_img_path_from_folder(valid_subjects)
    
    print('\tStep 4.\tGet which slice is unhealthy')
    unhealthy_train_slice = get_unhealthy_slice(train_list)
    unhealthy_valid_slice = get_unhealthy_slice(valid_list)
    
    print('\tStep 5.\tSlice 3D to 2D in prevision of torch.Dataset')
    train_data = Slice3Dto2D(train_list, unhealthy_train_slice, ratio_P_to_U)
    valid_data = Slice3Dto2D(valid_list, unhealthy_valid_slice, ratio_P_to_U)

    return train_data, valid_data