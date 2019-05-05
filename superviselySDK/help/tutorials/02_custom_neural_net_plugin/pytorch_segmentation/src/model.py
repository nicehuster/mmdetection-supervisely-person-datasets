# coding: utf-8

from torch import nn

HEAD = 'head'
INPUT_SIZE = 'input_size'
HEIGHT = 'height'
WIDTH = 'width'


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Preserves input spatial dimensions.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self._seq(inputs)


class PyTorchSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._layers = nn.Sequential(
            ConvBNAct(in_channels=3, out_channels=10),
            ConvBNAct(in_channels=10, out_channels=20),
        )
        self._head = nn.Conv2d(in_channels=20, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, inputs):
        return self._layers(inputs)
