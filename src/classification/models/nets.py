import yaml
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
cpath = Path().absolute().joinpath('config.yaml')
print(cpath)
config_file = Path(cpath)
with open(config_file) as file:
  config = yaml.safe_load(file)

class ResNet3D_18_Classifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, linear_ch=512, seed=None, early_layers_learning_rate=0):
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(ResNet3D_18_Classifier, self).__init__()
        if seed != None:
            print(f"Seed set to {seed}")
            torch.manual_seed(seed)
            
        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        if not early_layers_learning_rate: # 
            print("Freezing layers")
            for p in self.model.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            print(f"Early layers will use a learning rate of {early_layers_learning_rate}")
        #Reshape
        print(f"Initializing network for {in_ch} channel input")
        if in_ch!=3:
            self.model.stem[0] =  nn.Conv3d(in_ch, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model.fc = nn.Linear(linear_ch, out_ch)
        print(f"Linear layer initialized with {linear_ch} number of channels.")
        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet3D_18_Classifier, self).__init__(self.model,
                                                     self.out)

class ResNet18Classifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, seed=None, early_layers_learning_rate=0):
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(ResNet18Classifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # model.classifier[1]=nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization

        if not early_layers_learning_rate: # 
            print("Freezing layers")
            for p in self.model.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            print(f"Early layers will use a learning rate of {early_layers_learning_rate}")
        self.model.fc = nn.Linear(512, out_ch)

        if isinstance(self.model.fc, nn.Linear):
            torch.nn.init.xavier_uniform_(self.model.fc.weight)
            if self.model.fc.bias is not None:
                torch.nn.init.zeros_(self.model.fc.bias)

        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet18Classifier, self).__init__(self.model, 
                                                 self.out)
                                                
class SqueezeNetClassifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, seed=None, early_layers='freeze'):
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(SqueezeNetClassifier, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
        model.classifier[1]=nn.Conv2d(512, out_ch, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization
        if isinstance(model.classifier[1], nn.Conv2d):
            torch.nn.init.xavier_uniform_(model.classifier[1].weight)
            if model.classifier[1].bias is not None:
                torch.nn.init.zeros_(model.classifier[1].bias)
        
        super(SqueezeNetClassifier, self).__init__(self.model)