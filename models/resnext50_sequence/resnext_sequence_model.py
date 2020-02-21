import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Resnext50Features(nn.Module):
    def __init__(self):
        super(Resnext50Features, self).__init__()
        resnext50 = models.resnext50_32x4d(pretrained=True)
        resnext50.fc = Identity()
        self.resnext50 = resnext50
        
    def forward(self, x):
        
        x = self.resnext50(x)
        return x
    
class ResCNNEncoder(nn.Module):
    def __init__(self):
        super(ResCNNEncoder, self).__init__()
        
        self.net = Resnext50Features()
        
    def forward(self, x):
        x = self.net(x)
        
        return x
    
class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=2048,
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
    
class SequenceModelResnext(nn.Module):
    def __init__(self):
        super(SequenceModelResnext, self).__init__()

        self.encoder_model = ResCNNEncoder()
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