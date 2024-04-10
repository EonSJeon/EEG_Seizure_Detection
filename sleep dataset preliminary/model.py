import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_count, pool_type='max', kernel_size=3, pool_kernel_size=2, pool_stride=2, dropout_p=0.5):
        super(ConvBlock, self).__init__()
        layers = []

        for i in range(conv_count):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2))
            layers.append(nn.GELU())

        if pool_type == 'max':
            layers.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride))
        elif pool_type == 'avg':
            layers.append(nn.AvgPool2d(pool_kernel_size, stride=pool_stride))

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.Dropout(dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualConnection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class SleepStageCNN(nn.Module):
    def __init__(self):
        super(SleepStageCNN, self).__init__()
        self.block1 = ConvBlock(3, 32, conv_count=3, pool_type='max')
        self.res1 = ResidualConnection(3, 32, stride=2)
        self.block2 = ConvBlock(32, 32, conv_count=3, pool_type='avg', pool_kernel_size=3, pool_stride=2)
        self.block3 = ConvBlock(32, 64, conv_count=3, pool_type='max')
        self.res3 = ResidualConnection(32, 64, stride=2)
        self.block4 = ConvBlock(64, 64, conv_count=3, pool_type='avg', pool_kernel_size=3, pool_stride=2)
        self.block5 = ConvBlock(64, 64, conv_count=3, pool_type='avg', pool_stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity = self.res1(x)
        x = self.block1(x)
        x = self.block2(x + identity)

        identity = self.res3(x)
        x = self.block3(x)
        x = self.block4(x + identity)

        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x

# Example usage
model = SleepStageCNN()
print(model)

# Assuming input EEG spectrogram size is (batch_size, 3, 76, 60)
input = torch.randn(1, 3, 76, 60)
output = model(input)
print(output.shape)  # Expected output shape: (batch_size, 64)
