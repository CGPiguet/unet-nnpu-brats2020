import os
import urllib.request
import zipfile

def download_BraTS2020(destination_dir: str):
    """Download and unzip the BraTS2020 dataset into destination_dir

    Args:
        root_dir (str): PATH to where download and unzip
    """
    zip_file = destination_dir + '.zip'
    if not os.path.exists(destination_dir):
      if not os.path.exists(zip_file):
        urllib.request.urlretrieve('Please post the link to the training BraTS2020 dataset', zip_file)
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
          zip_ref.extractall()
                