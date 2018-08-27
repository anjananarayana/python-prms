import numpy as np
import pandas as pd
import pydot
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
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
        #convert the training data & testing data and labels  into dataframe
        train_df=pd.DataFrame(train_data)
        test_df=pd.DataFrame(test_data)
        train_labels_df=pd.DataFrame(train_labels)
        test_labels_df=pd.DataFrame(test_labels)
        print('Details of the input csv file: ',data.info())
        #Shape of the training data set
        print('Training Features Shape:', train_data.shape)
        # head of the training data
        print('Training dataset:',train_df.head())
        print('Training Labels Shape:', train_labels.shape)
        print('Train label data set (Benign (2) / Malignant (4):',train_labels_df.head())
        print('Testing Features Shape:', test_data.shape)
        print('Test feature data set:',test_df.head())
        print('Testing Labels Shape:', test_labels.shape)
        print('Test label data set:',test_labels_df.head())
        feature_list = list(train.columns)
        labels_list = tuple([str(i) for i in train_labels_df[train_labels_df.columns.tolist()[0]].unique()])
        #print("labels_list:",labels_list)
        #Save the files into local as csv
        print('Saving the train datset into local :',train_df.to_csv('training_data.csv'))
        print('Saving the train datset into local :',test_df.to_csv('test_data.csv'))
        print('Saving the train datset into local :',train_labels_df.to_csv('train_labels.csv'))
        print('Saving the train datset into local :',test_labels_df.to_csv('test_labels.csv'))
        #commute the correlation matrix for the input file
        corre = pd.DataFrame(data.corr())
        print('correlation values of the variables :',corre)
        sns.heatmap(corre,
        xticklabels=corre.columns,
        yticklabels=corre.columns)
        plt.show()

        # building the randomforest classifer on the train data and train lables by training the model
        rf_bc = RandomForestClassifier(n_estimators=10, max_depth=5)
        # Train the model on training data
        rf_bc.fit(train_data, train_labels)
        # Make Predictions on the Test Set
        # Use the forest's predict method on the test data
        predictions = rf_bc.predict(test_data)
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        #visulaization of the graph with output target classes
        tree_bc = rf_bc.estimators_[5]
        export_graphviz(tree_bc, out_file='tree_bc.dot', feature_names=feature_list,
                class_names=labels_list, rounded=True, precision=1)
        # Use dot file to create a graph
        (graph, ) = pydot.graph_from_dot_file('tree_bc.dot')
        # Write graph to a png file
        graph.write_png('tree_bc.png')

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
