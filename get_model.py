from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

models = {
    "densenet121": models.densenet121(pretrained=True),  # 1024
    "vgg16": models.vgg16(pretrained=True),  # 25088
    "alexnet": models.alexnet(pretrained=True)  # 9216
}

sizes = {
    "densenet121": {
        "in_size": 1024,
        "out_size": 102
    },
    "vgg16": {
        "in_size": 25088,
        "out_size": 102
    },
    "alexnet": {
        "in_size": 9216,
        "out_size": 102
    },
}


def get_model(arch, learning_rate, hidden_units, hidden_layers):
    """
    Takes arch type, learning rate (float) and number of hidden units (int)
    Returns a model, criterion, optimizer, input size, output size
    """
    model = models[arch]
    hidden_units = int(hidden_units)

    for param in model.parameters():
        param.requires_grad = False
    in_size = sizes[arch]["in_size"]
    out_size = sizes[arch]["out_size"]
    classifier = [nn.Linear(in_size, hidden_units),
                  nn.ReLU(),
                  nn.Dropout(0.2)]

    for _ in range(int(hidden_layers)):
        classifier.append(nn.Linear(hidden_units, hidden_units))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(0.2))

    classifier.append(nn.Linear(hidden_units, out_size))
    classifier.append(nn.LogSoftmax(dim=1))
    model.classifier = nn.Sequential(*classifier)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=float(learning_rate))
    return model, criterion, optimizer, in_size, out_size
