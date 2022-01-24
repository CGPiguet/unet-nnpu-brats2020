# nnpu-segmentation
 Segmentation on BraTS2020 dataset using non-negative Positive-Unlabeled loss function. 
## Motivation 
Segmentation is a task of high importance in medical image analysis. Its ability to provide fast and reliable delimitation of structure of interest has made it particularly useful in multiple domains. 
Unfortunately, deep learnings methods requires a large amount of annotated data. Theses annotaded data are costly and requires a lot of time to be produced. 
Therefore, the desing of new methods requiring a limited amount of annotated data has attracted the interest of the research community. 
In this case, the use of Positive-Unlabeled learning is explored. 
# nnPu loss 
The current state of the art of Positive-Unlabeled learning is the non-negative Risk estimator or nnPU loss. 
This project proposes a PyTorch implementation of the nnPu loss based on the original paper and its corresponding Chainer implementation. 
The implementation can be found in `nnPULoss.py` file. 
# Segmentation 
The model used for the segmentation is a standard U-Net that can be found in `model.py`. 
The dataset used is a conversion of the BraTS2020 dataset in 2D. 
## How to use ? 
### Training 
An exemple of how to launch the training is given in `run_train_exemple.sh`. Argparse module is used to pass arguments to the code to modify the training. 
All arguments can be found in `run_train.py`. The argument `--name` determines the name that will be used to create the folder in which the training parameters, results and model state_dict will be saved.
For exemple, if the argument is set to  `--name="Exemple"`, the parameters of the training, the results, and the state_dict of the model will be saved in a folder `model_saved_Exemple`.
Furthermore, to continue the training, the argument `-continue_training' must be set to `True`. Then the code will search the folder with the corresponding name argument and load the latest files that contains the state dict of the model and the optimizer. 
### Save Prediction 
Due to the great imbalance in the data, the initial descrimination threshold of 0 is not respected. So the file `run_ValidSavePred.py`, is used to save all the prediction of the Validation dataset model for then used the jupyter lab `DiceThreshold.ipynb` to determine the best discrimination threshold. 
### BraTS2020 download and conversion 
If the BraTS2020 dataset is not already present, the model will download it automatically and save it in the current foler under the name of 'MICCAI_BraTS2020_TrainingData'.
Then, if the converted 2D dataset is not in the file, the conversion will be automatically based on the 'RatioTrainValid' parameter which determines the percentage with which the dataset should be separated between the training set and the validation set.  And based on the 'RatioPosToNeg' parameter which determines the percentage of Positive samples that will be defined as Unlabeled to generate the sparse annotation used during the training.
The download and conversion of the BraTS2020 dataset can be use in a standalone version where an exemple can be found in `run_preprocess2D_standaloneExemple.sh`. 







