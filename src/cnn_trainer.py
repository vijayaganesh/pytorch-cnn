# This script generates the CNN Trainer class to batch-ify the mnist data and has
# methods to train and validate the data
__author__ = "VijayaGanesh Mohan"
__email__ = "vmohan2@ncsu.edu"


import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from configuration import Configuration
from network import ConvNetwork
from utils import convert_to_one_hot
from utils import convert_from_one_hot




class CNNTrainer:
    def __init__(self, mnist_data, train_ratio, config_file):
        unsplit_data, unsplit_label = mnist_data.load_training()
        unsplit_data = np.array([[CNNTrainer._reshape_data(x)] for x in unsplit_data])
        unsplit_label = np.array([x for x in unsplit_label])
        self.train_data, self.train_label, self.validate_data, self.validate_label = \
            CNNTrainer._split_training(unsplit_data, unsplit_label, train_ratio)
        self.configuration = Configuration(config_file)
        network = ConvNetwork(self.configuration)
        self.device = torch.device('cpu')
        self.model = network.to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = self.configuration.optimizer(self.model.parameters(), 
                                                      lr=self.configuration.learning_rate)
                
    def train(self):

        train_batch = DataLoader(TensorDataset(torch.from_numpy(self.train_data).float(),
                                               torch.from_numpy(self.train_label).long()), self.configuration.batch_size,
                                 shuffle=True)

        validation_data = DataLoader(TensorDataset(torch.from_numpy(self.validate_data).float(),
                                                   torch.from_numpy(self.validate_label).long()), shuffle=True)

        print("Steps per Epoch: "+str(len(train_batch)))
        for epoch in range(1, self.configuration.n_epoch+1):
            for step, data in enumerate(train_batch):   

                image = data[0].to(self.device)
                label = data[1].to(self.device)

                # Forward Propogation

                forward_output = self.model(image)
                loss = self.loss(forward_output, label)

                # Backward Propogation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (step+1) % self.configuration.batch_size == 0:
                    print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, self.configuration.n_epoch, step+1,
                                  len(train_batch), loss.item()))


        # Validation at the end of each epoch
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = len(self.validate_label)

            for image, label in validation_data:
                image = image.to(self.device)
                label = label.to(self.device)
                eval_output = np.array(convert_from_one_hot(self.model(image).numpy()))  
                if eval_output == label:
                    correct += 1

            print('Validation Accuracy: {}/{}'.format(correct, total))

    @staticmethod
    def _split_training(data, label, ratio):
        n_data = len(data)
        train_idx = random.sample(range(n_data), int(ratio * n_data))
        val_idx = np.setdiff1d(range(n_data), train_idx)
        return data[train_idx], label[train_idx], data[val_idx], label[val_idx]

    @staticmethod
    def _reshape_data(val):
        val_arr = np.array(val)
        width = int(np.sqrt(len(val_arr)))
        val_arr.shape = (width, width)
        return val_arr
        
    def evaluate(self):
        pass