#!/usr/bin/env python3

## This script defines the Neural Network Class
__author__ = "VijayaGanesh Mohan"
__email__ = "vmohan2@ncsu.edu"

import argparse
import os
import sys
import mnist

from network import ConvNetwork
from cnn_trainer import CNNTrainer


MNIST_DATA_PATH = "../data"
TRAIN_VALIDATE_RATIO = 0.7

def main(args):
    if os.path.exists(MNIST_DATA_PATH):
        mnist_data = mnist.MNIST(MNIST_DATA_PATH)
        trainer = CNNTrainer(mnist_data, TRAIN_VALIDATE_RATIO, args.config)
        trainer.train()

    else:
        print("MNIST data not found in the Data/ directory. Add the MNIST data to the path: ")
        print(os.path.join(os.getcwd(), '..', 'data'))

        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on top of the MNIST Data.")
    parser.add_argument('-p', '--config', help="Path of the Configuration YAML file")   
    args = parser.parse_args()
    main(args)
