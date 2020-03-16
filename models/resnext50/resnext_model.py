import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Resnext50(nn.Module):
    def __init__(self, model_name = '', isPretrained=True, freeze_bn_affine=False, freeze_bn=False):
        super(Resnext50, self).__init__()
        resnext50 = models.resnext50_32x4d(pretrained=isPretrained)
        resnext50.fc = nn.Linear(2048,1)
        self.resnext50 = resnext50
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Resnext50, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.efficient_net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        
    def forward(self, x):
        x = self.resnext50(x)
        return x
    
    