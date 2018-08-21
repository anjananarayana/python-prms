import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow import keras

from sklearn.model_selection import train_test_split
# reading the dataset
import csv
import sys


print('sys.argv is', sys.argv)
try:
    if (sys.argv[1] == '-'):
        f = sys.stdin.read().splitlines()
    else:
        filename = sys.argv[1]
        f = open(filename, 'r')
    csv = csv.reader(f)
    data = list(csv)
    for row in data:
        print(row)
except Exception as e:
    print("Error Reading from file:")
