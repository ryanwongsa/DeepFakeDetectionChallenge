import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import torch

class FeatureModel(nn.Module):
    
    def __init__(self, num_features = 512):
        super(FeatureModel, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
        self.layers = layers = {
            '3':"relu1_2",
            '8':"relu2_2",
            '15':"relu3_3",
            '22':"relu4_3",
        }
        
        self.conv1 = nn.Conv2d(64, num_features, 3, 1, (1,1))
        self.conv1_bn = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(128, num_features//2, 3, 1, (1,1))
        self.conv2_bn = nn.BatchNorm2d(num_features//2)
        self.conv3 = nn.Conv2d(256, num_features//4, 3, 1, (1,1))
        self.conv3_bn = nn.BatchNorm2d(num_features//4)
        self.conv4 = nn.Conv2d(512, num_features//8, 3, 1, (1,1))
        self.conv4_bn = nn.BatchNorm2d(num_features//8)
        
        self.pool2x2 = nn.AvgPool2d(2, 2)
        self.pool4x4 = nn.AvgPool2d(4, 4)
        self.pool8x8 = nn.AvgPool2d(8, 8)
        
        self.total_convs = num_features + num_features//2 + num_features//4 + num_features//8
        self.conv5 = nn.Conv2d(self.total_convs, self.total_convs, 3, 1, (1,1))
        self.conv5_bn = nn.BatchNorm2d(self.total_convs)
        
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, images):
        features = []
        x = images
        bs = x.size(0)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                if name == '3':
                    x_res = self.conv1_bn(self.conv1(x))
                    x_res = F.relu(self.pool8x8(x_res))
                if name == '8':
                    x_res = self.conv2_bn(self.conv2(x))
                    x_res = F.relu(self.pool4x4(x_res))
                if name == '15':
                    x_res = self.conv3_bn(self.conv3(x))
                    x_res = F.relu(self.pool2x2(x_res))
                if name == '22':
                    x_res = F.relu(self.conv4_bn(self.conv4(x)))
                
                features.append(x_res)
        features = torch.cat(features,1)
        features = self._avg_pooling(self.conv5_bn(self.conv5(features)))
        
        features = features.view(bs, -1)
        return features