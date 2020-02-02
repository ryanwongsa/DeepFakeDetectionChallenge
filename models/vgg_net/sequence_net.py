import torch.nn as nn
import torch.nn.functional as F
from models.vgg_net.decoder_rnn import DecoderRNN
import torch
from models.vgg_net.feature_model import FeatureModel

class SequenceNet(nn.Module):
    def __init__(self, features = 256):
        super(SequenceNet, self).__init__()
        self.FeatureModel = FeatureModel(features)
        self.embed_dim = self.FeatureModel.total_convs
   
        self.out_act = nn.Sigmoid()
        
        fc_hidden1, fc_hidden2, h_RNN_layers, h_RNN, h_FC_dim = 512, 512, 2, 512, 128
        
        self.rnn_decoder = DecoderRNN(CNN_embed_dim=self.embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim, drop_p=0.2, num_classes=1)


    def forward(self, x_3d):
        result = []
        for t in range(x_3d.size(1)):
            x = self.FeatureModel(x_3d[:, t, :, :, :])
            result.append(x)
            
        x = torch.stack(result, dim=0).transpose_(0, 1)
        x = self.rnn_decoder(x)
        return x