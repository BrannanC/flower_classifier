import numpy as np
import torch
from PIL import Image
import json

from get_input_args import get_predict_args
from get_model import get_model


def main():
    predict_args = get_predict_args()
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    if predict_args.gpu and not torch.cuda.is_available():
        raise Exception('GPU not available')

    c = torch.load(predict_args.checkpoint[0])
    model, criterion, optimizer, in_size, out_size = get_model(
        c['arch'], c['learning_rate'], c['hidden_units'], c['hidden_layers'])
    model.load_state_dict(c['state_dict'])

    probs, flowers = predict(
        predict_args.filepath[0], model, int(predict_args.topk), predict_args.gpu, c['class_to_idx'], cat_to_name)

    print('\nRESULTS:')
    for i, p in enumerate(probs):
        print("    " + flowers[i] + ": " + " {:.2f}".format(p * 100) + "%")


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor
    '''
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin,
                        top_margin))
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    tensor = tensor.float()
    return tensor


def predict(image_path, model, topk, gpu, class_to_idx, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    img = process_image(img)
    img.unsqueeze_(0)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and gpu else "cpu")
    model = model.to(device)
    img = img.to(device)
    model.eval()
    res = model(img)
    probs = torch.exp(res)
    probs, labels = probs.topk(topk)
    ls = labels.cpu().detach().numpy().tolist()[0]
    probs = probs.cpu().detach().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in
                    class_to_idx.items()}
    labels = [idx_to_class[lab] for lab in ls]
    flowers = [cat_to_name[idx_to_class[l]] for l in ls]
    return probs, flowers


if __name__ == "__main__":
    main()
