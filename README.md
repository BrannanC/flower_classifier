# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program.


## train.py
### Classifies flower images using a pretrained CNN model
##### Command Line Arguments:
  - `--dir`, Image Folder as --dir with default value 'flower_images'
  - `--arch`, CNN Model Architecture as --arch with default value 'densenet121 options are 'densenet121', 'vgg16' and 'alexnet'
  - `--hidden_units`, Hidden Units as --hidden_units with default value '512'
  - `--hidden_layers`, Hidden Layers as --hidden_layers with default value '1'
  - `--learning_rate`, Learning Rate as --learning_rate with default value '0.005'
  - `--epochs`, Epochs as --epochs with default value '10
  - `--gpu`, Flag, GPU as --gpu defaults to False unless set
  - `--save_dir`, Directory to save checkpoint in as --save_dir with default 'checkpoints/'
    
    
## predict.py
### Predicts a flower name from and image using a checkpoint
##### Command Line Arguments:
  - Filepath of image to be predicted
  - Checkpoint to be used in prediction
  - `--topk`, Number of Top Predictions as --topk with default value '3
  - `--category_names`, Category Names as --category_names with default value 'cat_to_name.json'
  - `--gpu`, GPU as --gpu defaults to False unless set
