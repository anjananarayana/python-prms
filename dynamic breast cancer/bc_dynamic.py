import numpy as np
import pandas as pd
import pydot
from sklearn import tree
import matplotlib.pyplot as plt
#import plotly.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import argparse


class Test(object):
    def split_data(self, input_file, target_variable):
        data = pd.read_csv(input_file)
        train = data.drop(target_variable, axis=1)
        test = data[target_variable]
        train_data, test_data, train_labels, test_labels = train_test_split(
            train, test, test_size=1/3, random_state=5)
        print('Training Features Shape:', train_data.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_data.shape)
        print('Testing Labels Shape:', test_labels.shape)
        rf_bc = RandomForestClassifier(n_estimators=1000, random_state=42)
        # Train the model on training data
        rf_bc.fit(train_data, train_labels)
        # Make Predictions on the Test Set
        # Use the forest's predict method on the test data
        predictions = rf_bc.predict(test_data)
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group()
    required_args.add_argument(
        "-i", "--input-file", dest="input_file", required=True)
    required_args.add_argument(
        "-t", "--target_variable", dest="target_variable", required=True
    )
    arguments = parser.parse_args()
    tr_obj = Test()
    tr_obj.split_data(arguments.input_file,
                      arguments.target_variable)
