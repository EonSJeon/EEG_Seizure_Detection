

# block1
# {
#     Conv(16,3*3,1)
#     Conv(32,3*3,1)
#     Conv(32,3*3,1)
#     Max-pool(2*2, 2)
#     Batch Normalization
#     Drop-out(p=0.5)
# }

# block2
# {
#     Conv(16,3*3,1)
#     Conv(32,3*3,1)
#     Conv(32,3*3,1)
#     Avg-pool(2*2, 2)
#     Batch Normalization
#     Drop-out(p=0.5)
# }

# block3
# {
#     Conv(20,1*1,1)
#     Conv(64,3*3,1)
#     Conv(64,3*3,1)
#     Max-pool(2*2, 2)
#     Batch Normalization
#     Drop-out(p=0.5)
# }

# block4
# {
#     Conv(20,1*1,1)
#     Conv(32,3*3,1)
#     Conv(64,3*3,1)
#     Avg-pool(2*2, 2)
#     Batch Normalization
#     Drop-out(p=0.5)
# }

# block5
# {
#     Conv(20,1*1,1)
#     Conv(64,3*3,1)
#     Conv(64,3*3,1)
#     Max-pool(2*2, 2)
#     Batch Normalization
#     Drop-out(p=0.5)
# }

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool, pool_type='max', dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2)

        if pool_type == 'max':
            self.pool = nn.MaxPool2d(pool, stride=pool[1])
        else:
            self.pool = nn.AvgPool2d(pool, stride=pool[1])

        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class SleepStageCNN(nn.Module):
    def __init__(self):
        super(SleepStageCNN, self).__init__()
        self.block1 = ConvBlock(18, 32, 3, 1, (2, 2), 'max', 0.5)
        self.block2 = ConvBlock(32, 32, 3, 1, (2, 2), 'avg', 0.5)
        self.block3 = ConvBlock(32, 64, 3, 1, (2, 2), 'max', 0.5)
        self.block4 = ConvBlock(64, 64, 3, 1, (2, 2), 'avg', 0.5)
        self.block5 = ConvBlock(64, 64, 3, 1, (2, 2), 'max', 0.5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        identity1 = x
        x = self.block2(x)


        identity1 = F.interpolate(identity1, size=(x.size(2), x.size(3)), mode='nearest')
        x += identity1  # Residual connection after block 2

        x = self.block3(x)
        identity2 = x
        x = self.block4(x)


        identity2 = F.interpolate(identity2, size=(x.size(2), x.size(3)), mode='nearest')
        x += identity2  # Residual connection after block 4

        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class EEGSNet(nn.Module):
    def __init__(self):
        super(EEGSNet, self).__init__()
        self.cnn = SleepStageCNN()
        self.bilstm = BiLSTM(input_size=64, hidden_size=128, num_layers=2, num_classes=4)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.shape
        # Process each image in the sequence through CNN
        cnn_out = torch.tensor([]).to(x.device)
        for i in range(seq_length):
            out = self.cnn(x[:, i, :, :, :])  # Process each image through the CNN
            cnn_out = torch.cat((cnn_out, out.unsqueeze(1)), dim=1)

        # Now cnn_out should be [batch_size, seq_length, features]
        output = self.bilstm(cnn_out)
        return output

# Assuming your input is a batch of images treated as a sequence
input_tensor = torch.randn(1, 3, 18, 129, 111)  # 1 batch, 10 sequence length, 18 channels, 129x111 spatial dimensions
model = EEGSNet()
output = model(input_tensor)
print(output.shape)  # Expected shape: [1, 4] for classification

