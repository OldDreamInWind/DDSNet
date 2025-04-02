import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU()
                                       )
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU()
                                       )
    def forward(self, x):
        x=self.depthwise(x)
        x=self.pointwise(x)
        return x

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(DepthwiseSeparableConv(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Connect input and output in each channel
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
class DDSNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.b1 = nn.Sequential(
                                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # output of last dense block
            num_channels += num_convs * growth_rate
            # Add transition block between dense blocks, half the channels
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net = nn.Sequential(self.b1, *blks,
                                nn.BatchNorm2d(num_channels), nn.ReLU(),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(num_channels, output_channels))

    def forward(self, x):
        x = self.net(x)
        return x