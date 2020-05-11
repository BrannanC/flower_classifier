import argparse


def get_training_args():
    """
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flower_images'
      2. CNN Model Architecture as --arch with default value 'densenet121 options are 'densenet121', 'vgg16' and 'alexnet'
      3. Hidden Units as --hidden_units with default value '512'
      4. Hidden Layers as --hidden_layers with default value '1'
      5. Learning Rate as --learning_rate with default value '0.005'
      6. Epochs as --epochs with default value '10
      7. GPU as --gpu defaults to False unless set
      8. Directory to save checkpoint in as --save_dir with default 'checkpoints/'
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(
        description='Classifies flower images using a pretrained CNN model')
    parser.add_argument('--dir', type=str, default='flower_data',
                        help='Directory containing images')
    parser.add_argument('--arch', type=str,
                        default='densenet121', help='Model')
    parser.add_argument('--hidden_units', type=str,
                        default='512', help='Hidden Units for Model')
    parser.add_argument('--hidden_layers', type=str,
                        default='1', help='Hidden Layers for Model')
    parser.add_argument('--learning_rate', type=str,
                        default='0.005', help='Learning Rate for Model')
    parser.add_argument('--epochs', type=str,
                        default='10', help='Number of epochs to train model')
    parser.add_argument("--gpu",
                        action="store_true",
                        help="Sets use of GPU")
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='Directory to save checkpoint')
    return parser.parse_args()


def get_predict_args():
    """
    Command Line Arguments:
      1. Filepath of image to be predicted
      2. Checkpoint to be used in prediction
      3. Number of Top Predictions as --topk with default value '3
      4. Category Names as --category_names with default value 'cat_to_name.json'
      5. GPU as --gpu defaults to False unless set
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(
        description='Classifies flower images using a pretrained CNN model')
    parser.add_argument('filepath', type=str,
                        nargs=1, help='Filepath of image to predict')
    parser.add_argument('checkpoint', type=str,
                        nargs=1, help='Filepath of image to predict')
    parser.add_argument('--topk', type=str,
                        default='3', help='Number of classes to display in prediction')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        nargs=1, help='Filepath of image to predict')
    parser.add_argument("--gpu",
                        action="store_true",
                        help="Sets use of GPU")
    return parser.parse_args()
