import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet_sequences.model import EfficientNet
from models.efficientnet_sequences.utils import round_filters
import torch

class SequenceNet(nn.Module):
    def __init__(self, model_name = 'efficientnet-b0', isPretrained=True):
        super(SequenceNet, self).__init__()
        if isPretrained==True:
            self.efficient_net = EfficientNet.from_pretrained(model_name)
        else:
            self.efficient_net = EfficientNet.from_name(model_name)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.efficient_net._global_params.dropout_rate)

        out_channels = round_filters(1280, self.efficient_net._global_params)
        self._fc = nn.Linear(out_channels, 1)

        self.out_act = nn.Sigmoid()

    def forward(self, x_3d, model_types = [0]):
        result = []
        for t in range(x_3d.size(1)):
            x = self.efficient_net.extract_features(x_3d[:, t, :, :, :])
        
            x = self._avg_pooling(x)
            
            x = x.view(x_3d.size(0), -1)
            x = self._dropout(x)
            x = self._fc(x)
            x = self.out_act(x)
            result.append(x)
        return torch.stack(result, dim=0).transpose_(0, 1)