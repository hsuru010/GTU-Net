

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torchvision import transforms

from utils.functions import custom_transform
from models.ADT_SmaAt_UNet import SmaAt_UNet_ADT
from models.ADT_SST_SmaAt_UNet import SmaAt_UNet_ADT_SST
from models.SST_SmaAt_UNet import SmaAt_UNet_SST


"""
Only modify this file to make changes to forecasting process.

Dataloader config:
 - sequence_length variable determines the prediction horizon
 
Model config:
 - model_name determines the model being used 
   (MAKE SURE to change include_sst to False in dataloader_config if using ADT-SmaAt-UNet)
"""

#---------------------------------------------#
#             DATALOADER CONFIG               #
#---------------------------------------------#

transform = transforms.Compose([
    transforms.Lambda(custom_transform)
])

"""Define prediction horizon for forecasts.
"""
####################
sequence_length = 7
####################

dataloader_config={'dataset_path': '...\data\\adt_1200_00_23_filled.npy',
                   'include_sst': False, # set to False if using the ADT-SmaAt-UNet without SST data
                   'length_source': sequence_length, # how many timestep in inputs
                   'length_target': sequence_length, # how many timestep for prediction
                   'timestep': 1, # days between each inputs
                   'transform': transform,
                   'valid_ratio': 0.10, # ratio to use for validation and test sets
                   'batch_size': 16,
                   'small_train': False,
                   'model_is_3dconv': False, # in case using a model with 3D convolutions; rare
                   'scale_with_train': False} # in case want to standard scale all datasets using the mean/std of the training set; rare



#---------------------------------------------#
#                MODEL CONFIG                 #
#---------------------------------------------#

models_dict = {
    "smaat_unet_adt": SmaAt_UNet_ADT,
    "smaat_unet_sst": SmaAt_UNet_SST,
    "smaat_unet_adt_sst": SmaAt_UNet_ADT_SST
}

"""Select model to be used. 
If using an SST data-based model, include_sst must be set to True in the dataloader_config.
"""
#################################
model_name = 'smaat_unet_adt'
#################################

if model_name == 'smaat_unet_adt':
    model_params = {'n_channels': sequence_length, 'n_classes': sequence_length}
elif model_name == 'smaat_unet_sst':
    model_params = {'n_channels': sequence_length, 'n_classes': sequence_length}
elif model_name == 'smaat_unet_adt_sst':
    model_params = {'n_channels': sequence_length*2, 'n_classes': sequence_length*2}