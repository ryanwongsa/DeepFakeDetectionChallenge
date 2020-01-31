import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet_sequences.model import EfficientNet
from models.efficientnet_sequences.utils import round_filters
from models.efficientnet_sequences.decoder_rnn import DecoderRNN

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
        
        self.rnn_decoder = DecoderRNN(CNN_embed_dim=300, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=1)
        
        fc_hidden1, fc_hidden2, embed_dim = 512, 512, 300
        
        self.fc1 = nn.Linear(1280, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, embed_dim)

    def forward(self, x_3d, model_types = [1]):
        result_x0, result_x1 = [], []
        for t in range(x_3d.size(1)):
            x = self.efficient_net.extract_features(x_3d[:, t, :, :, :])
            x = self._avg_pooling(x)
            x = x.view(x_3d.size(0), -1)
            
            if 1 in model_types:
                x_1 = self.bn1(self.fc1(x))
                x_1 = F.relu(x_1)
                x_1 = self.bn2(self.fc2(x_1))
                x_1 = F.relu(x_1)
                x_1 = self._dropout(x_1)
                x_1 = self.fc3(x_1)
                result_x1.append(x_1)
                
            if 0 in model_types:
                x_0 = self._dropout(x)
                x_0 = self._fc(x_0)
                x_0 = self.out_act(x_0)

                result_x0.append(x_0)
                
        if 0 in model_types:
            x_0 = torch.stack(result_x0, dim=0).transpose_(0, 1)
            print(x_0.shape)
            return x_0
        
        if 1 in model_types:
            x_1 = torch.stack(result_x1, dim=0).transpose_(0, 1)
            x_1 = self.rnn_decoder(x_1)
            return x_1
        
        
        