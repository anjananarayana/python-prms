# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

bc=pd.read_csv("/home/anjana/anjana/csv/breast-cancer-wisconsin.csv")
# to see the cloumn names of the dataset
list(bc)
list(bc.columns.values)
import sklearn
print (sklearn.__version__)
#Features and Targets and Convert Data to Arrays
# Labels are the values we want to predict
# Labels are the values we want to predict
labels = np.array(bc['CancerType'])
# Remove the labels from the features
# axis 1 refers to the columns
bc= bc.drop('CancerType', axis = 1)
# Saving feature names for later use
feature_list = list(bc.columns)
# Convert to numpy array
features = np.array(bc)

#Training and Testing Sets

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
# to check the length of the dataset

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
#Establish Baseline
# The baseline predictions are the historical averages
baseline_preds_clump = test_features[:, feature_list.index('ClumpThickness')]
baseline_preds_unisize = test_features[:, feature_list.index('UniformityCellSize')]
baseline_preds_unishape = test_features[:, feature_list.index('UniformityCellShape')]
baseline_preds_marg = test_features[:, feature_list.index('MarginalAdhesion')]
baseline_preds_epi = test_features[:, feature_list.index('SingleEpithelialCellSize')]
baseline_preds_nuc = test_features[:, feature_list.index('BareNuclei')]
baseline_preds_bland = test_features[:, feature_list.index('BlandChromatin')]
baseline_preds_nor = test_features[:, feature_list.index('NormalNucleoli')]
baseline_preds_mit = test_features[:, feature_list.index('Mitoses')]

# Baseline errors, and display average baseline error
baseline_errors_clump = abs(baseline_preds_clump - test_labels)
baseline_errors_unisize = abs(baseline_preds_unisize - test_labels)
baseline_errors_unishape = abs(baseline_preds_unishape - test_labels)
baseline_errors_marg = abs(baseline_preds_marg - test_labels)
baseline_errors_epi = abs(baseline_preds_epi - test_labels)
baseline_errors_nuc = abs(baseline_preds_nuc - test_labels)
baseline_errors_bland = abs(baseline_preds_bland - test_labels)
baseline_errors_nor= abs(baseline_preds_nor - test_labels)
baseline_errors_mit= abs(baseline_preds_mit - test_labels)

a=round(np.mean(baseline_errors_clump), 2)
c=round(np.mean(baseline_errors_unisize), 2)
d=round(np.mean(baseline_errors_unishape), 2)
e=round(np.mean(baseline_errors_marg), 2)
f=round(np.mean(baseline_errors_epi), 2)
g=round(np.mean(baseline_errors_nuc), 2)
h=round(np.mean(baseline_errors_bland), 2)
i=round(np.mean(baseline_errors_nor), 2)
j=round(np.mean(baseline_errors_mit), 2)
y=[a,c,d,e,f,g,h,i,j]
label=['clump','unisize','unishape','marg','epi','nuc','bland','nor','mit']
#to plot the average base errors of the features
freq_series = pd.Series.from_array(y)
# Plot the figure.
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar')
ax.set_title(' The baseline predictions averages')
ax.set_xlabel('Features of the breast cancer')
ax.set_ylabel('Average errors')
ax.set_xticklabels(label)

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    label = "{:.1f}".format(y_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.

plt.savefig("image.png")
#Train Model

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

#Make Predictions on the Test Set
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


#variable importance
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importancesmportances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

## New random forest with only the two most important variables
rf_most_important = RandomForestClassifier(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('UniformityCellSize'), feature_list.index('BareNuclei')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
#%matplotlib inline
# Set the style



# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');





#link for this
#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/
