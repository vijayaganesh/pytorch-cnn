# This script parses the config.yaml file to generate the configuration object
__author__ = "VijayaGanesh Mohan"
__email__ = "vmohan2@ncsu.edu"

import inspect
import os
import torch
import yaml


class Configuration:
    def __init__(self, config_file):
        self._yaml_config = None
        if os.path.isfile(config_file):
            with open(config_file, 'r') as yaml_file:
                try:
                    self._yaml_config = yaml.load(yaml_file)
                except yaml.YAMLError as error:
                    print("Config File is corrupt. loading default")
                    print(error)
        try:
            self._parse_yaml()
        except KeyError as error:
            print(error)
            print("Corrupt YAML File, loading default")
            self._load_default_config()

        self.validate_config()

    def _load_default_config(self):
        self.n_epoch = 50
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.conv_stride = 1
        self.conv_padding = 2
        self.conv_kernel_size = 5
        self.max_pool_stride = 2
        self.max_pool_kernel_size = 4
        self.n_conv_layers = 1
        self.layers = [Layer(1, 5, True)]
        self.img_size = 28
        self.activation = torch.nn.modules.activation.ReLU
        self.n_classes = 10
        self.n_final_neurons = 20
        self.optimizer = torch.optim.Adam

    def _parse_yaml(self):
        if self._yaml_config is not None:
            try:
                self.n_epoch = self._yaml_config['n_epoch']
                self.batch_size = self._yaml_config['batch_size']
                self.learning_rate = float(self._yaml_config['learning_rate'])
                self.conv_stride = self._yaml_config['conv_stride']
                self.conv_padding = self._yaml_config['conv_padding']
                self.conv_kernel_size = self._yaml_config['conv_kernel_size']
                self.max_pool_stride = self._yaml_config['max_pool_stride']
                self.max_pool_kernel_size = self._yaml_config['max_pool_kernel_size']
                self.n_conv_layers = self._yaml_config['n_conv_layers']
                self.img_size = self._yaml_config['img_size']
                self.layers = [None] * self.n_conv_layers
                for i in range(1, self.n_conv_layers+1):
                    self.layers[i-1] = Layer(1,
                                        self._yaml_config['layers'][i]['n_kernels'],
                                        self._yaml_config['layers'][i]['is_batch_norm'])
                self.n_classes = self._yaml_config['n_classes']
                self.n_final_neurons = self._yaml_config['n_final_neurons']
                activation = self._yaml_config['activation']
                if activation in dict(inspect.getmembers(torch.nn.modules.activation)).keys():
                    self.activation = dict(inspect.getmembers(torch.nn.modules.activation)) \
                                                                                .get(activation)
                else:
                    self.activation = torch.nn.modules.activation.ReLU
                optimizer = self._yaml_config['optimizer']
                if optimizer in dict(inspect.getmembers(torch.optim)).keys():
                    self.optimizer = dict(inspect.getmembers(torch.optim)).get(optimizer)
                else:
                    self.optimizer = torch.optim.Adam
            except KeyError as error:
                print(error)
                raise KeyError("Invalid Key Found")
        else:
            self._load_default_config()

    def validate_config(self):
        # Checking the input channels of the convolutional layers
        prev_conv_channels = 0
        fc_size = self.img_size
        for i in range(self.n_conv_layers):
            if i == 0:
                self.layers[i].in_channels = 1
            else:
                self.layers[i].in_channels = prev_conv_channels
            prev_conv_channels = self.layers[i].out_channels
            fc_size = self._calc_conv_output_size(fc_size)
            print(fc_size)
            fc_size = self._calc_max_pool_output_size(fc_size)
            print(fc_size)

        self.fc_layer = Layer(int(fc_size * fc_size * prev_conv_channels), self.n_classes)

    def _calc_conv_output_size(self, w_in):
        return (w_in - self.conv_kernel_size + 2 * self.conv_padding) / self.conv_stride + 1

    def _calc_max_pool_output_size(self, w_in):
        return (w_in - self.max_pool_kernel_size) / self.max_pool_stride + 1


class Layer():
    def __init__(self, *args):
        self.in_channels = None
        self.out_channels = None
        self.is_batch_norm = False

        if args:
            try:
                self.in_channels = args[0]
                self.out_channels = args[1]
                self.is_batch_norm = args[2]
            except IndexError:
                pass
            