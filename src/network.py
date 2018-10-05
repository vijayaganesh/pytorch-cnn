## This script defines the Neural Network Class
__author__ = "VijayaGanesh Mohan"
__email__ = "vmohan2@ncsu.edu"

try:
    import sys
    import torch

except ImportError as error:
    print("Required module not found.")
    print("Please run pip3 install -r requirements.txt")
    print(error)
    sys.exit(1)

class ConvNetwork(torch.nn.Module):
    def __init__(self, config):

        super(ConvNetwork, self).__init__()
        self.configuration = config
        self.layers = [None]*self.configuration.n_conv_layers
        for i in range(self.configuration.n_conv_layers):
            if self.configuration.layers[i].is_batch_norm:
                self.layers[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.configuration.layers[i].in_channels,
                                    out_channels=self.configuration.layers[i].out_channels,
                                    kernel_size=self.configuration.conv_kernel_size,
                                    stride=self.configuration.conv_stride,
                                    padding=self.configuration.conv_padding),
                    torch.nn.BatchNorm2d(self.configuration.layers[i].out_channels),
                    torch.nn.MaxPool2d(kernel_size=self.configuration.max_pool_kernel_size,
                                       stride=self.configuration.max_pool_stride))
            else:
                self.layers[i] = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.configuration.layers[i].in_channels,
                                    out_channels=self.configuration.layers[i].out_channels,
                                    kernel_size=self.configuration.conv_kernal_size,
                                    stride=self.configuration.conv_stride,
                                    padding=self.configuration.conv_padding),
                    torch.nn.MaxPool2d(kernel_size=self.configuration.max_pool_kernel_size,
                                       stride=self.configuration.max_pool_stride))
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.configuration.fc_layer.in_channels,
                            out_features=self.configuration.fc_layer.out_channels),
            torch.nn.modules.activation.Softmax())

    def forward(self, input):
        out = input
        for i in range(self.configuration.n_conv_layers):
            out = self.layers[i](out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layer(out)
        return out
