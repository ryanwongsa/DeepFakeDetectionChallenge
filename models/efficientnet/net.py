import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet.model import EfficientNet
from models.efficientnet.utils import round_filters

class Net(nn.Module):
    def __init__(self, model_name = 'efficientnet-b0'):
        super(Net, self).__init__()

        self.efficient_net = EfficientNet.from_pretrained(model_name)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.efficient_net._global_params.dropout_rate)

        out_channels = round_filters(1280, self.efficient_net._global_params)
        self._fc = nn.Linear(out_channels, 1)

        self.out_act = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        x = self.efficient_net.extract_features(x)

        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        x = self.out_act(x)
        return x