# Utility scripts with helper methods for the project 3
__author__ = "VijayaGanesh Mohan"
__email__ = "vmohan2@ncsu.edu"

import numpy as np

def convert_to_one_hot(label, n_labels=10):
    return np.eye(n_labels)[label]

def convert_from_one_hot(one_hot_array):
    return np.where(one_hot_array == max(one_hot_array))[0][0]