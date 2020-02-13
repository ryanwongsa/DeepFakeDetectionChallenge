import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from models.efficientnet.net import Net

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class EfficientFeatures(nn.Module):
    def __init__(self, model_dir):
        super(EfficientFeatures, self).__init__()
        self.model = Net('efficientnet-b6',False)
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['model'])
        self.model._fc = Identity()
        
    def forward(self, x):
        
        x = self.model(x)
        return x
    
class ResCNNEncoder(nn.Module):
    def __init__(self, model_dir):
        super(ResCNNEncoder, self).__init__()
        
        self.net = EfficientFeatures(model_dir)
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=2304,
            hidden_size=512,
            num_layers=2,
            dropout=0.0,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

#         self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        
#         self.LSTM.flatten_parameters()
        x, (h_n, h_c) = self.LSTM(x, None)
        x = self.fc1(x)
        
    
#         x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.fc2(x)

        return x[:, -1, :]
    
class SequenceModelEfficientNet(nn.Module):
    def __init__(self, model_dir):
        super(SequenceModelEfficientNet, self).__init__()

        self.encoder_model = ResCNNEncoder(model_dir)
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.decoder_model = DecoderRNN()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        with torch.no_grad():
            x = x.view(batch_size * timesteps, C, H, W)
            x = self.encoder_model(x)
        x = x.view(batch_size, timesteps, -1)
        x = self.decoder_model(x)
        return x