from resnet import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os

def create_tranfer(num_classes, model_name, PATH, verbose=False):
    # load the model and custom model
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, num_classes)
        model_new = ResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(512, num_classes)
        model_new = ResNet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, num_classes)
        model_new = ResNet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(2048, num_classes)
        model_new = ResNet101(num_classes=num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(2048, num_classes) 
        model_new = ResNet152(num_classes=num_classes)
    else:
        raise ValueError('Model name not found')

    # weights visualization
    if verbose:
        dicts = model.state_dict()
        dicts_new = model_new.state_dict()
        for key in dicts.keys():
                print(key)

        print("---------------------------------------------------------------------")    
        for key in dicts_new.keys():
            print(key)

        print("---------------------------------------------------------------------")
        #zip the two dictionaries and show both the keys
        for key, key2 in zip(dicts.keys(), dicts_new.keys()):
            print(key,"  ---------   ", key2)
    #save the weights of the torchvision model
    torch.save(model.state_dict(), PATH)
    #load the weights of the torchvision model on the custom model
    model_new.load_state_dict(torch.load(PATH), strict=False)
    #save the weights of the custom model
    torch.save(model_new.state_dict(), PATH)


if __name__ == "__main__":
    num_classes = 50
    model_name = 'resnet50'
    PATH = 'models/'
    # create path if not exists
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    MODEL_NAME = 'custom_resnet50.pt'    
    PATH = PATH + MODEL_NAME
    create_tranfer(num_classes, model_name, PATH, verbose=False)