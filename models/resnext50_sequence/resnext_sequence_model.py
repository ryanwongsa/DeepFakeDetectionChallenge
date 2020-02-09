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
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.net(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
    
class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=2048,
            hidden_size=2048,
            num_layers=2,
            dropout=0.5,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x
    
class SequenceModelResnext(nn.Module):
    def __init__(self):
        super(SequenceModelResnext, self).__init__()

        self.encoder_model = ResCNNEncoder()
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.decoder_model = DecoderRNN()

    def forward(self, x):
        x = self.encoder_model(x)
        x = self.decoder_model(x)
        return x