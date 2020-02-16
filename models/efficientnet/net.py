import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet.model import EfficientNet
from models.efficientnet.utils import round_filters

class Net(nn.Module):
    def __init__(self, model_name = 'efficientnet-b0', isPretrained=True, freeze_bn_affine=False, freeze_bn=False):
        super(Net, self).__init__()
        if isPretrained==True:
            self.efficient_net = EfficientNet.from_pretrained(model_name)
        else:
            self.efficient_net = EfficientNet.from_name(model_name)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.efficient_net._global_params.dropout_rate)

        out_channels = round_filters(1280, self.efficient_net._global_params)
        self._fc = nn.Linear(out_channels, 1)

        self.out_act = nn.Sigmoid()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Net, self).train(mode)
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
        bs = x.size(0)
        x = self.efficient_net.extract_features(x)

        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        # x = self.out_act(x)
        return x