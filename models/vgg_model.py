import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, name, nc):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], name)
        
        if name == "caltech101":
            in_feature = 8*64*7*7
        elif name == "stl10":
            in_feature = 8*64*3*3
        else:
            in_feature = 8*64*1*1
            
        #self.classifier = nn.Linear(in_features=in_feature, out_features=nc)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=in_feature, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=nc),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        #out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, d):
        layers = []
        in_channels = 3
            
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)