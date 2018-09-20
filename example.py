import pandas as pd
import numpy as np
import pdb

# data = pd.read_csv("/home/anjana/anjana/csv/combined_transactions.csv")

data = pd.read_csv("/home/anjana/Desktop/short.csv")


# def outliers_z_score(ys):
#     threshold = 3

#     mean_y = np.mean(ys)
#     stdev_y = np.std(ys)
#     z_scores = [(y - mean_y) / stdev_y for y in ys]
#     return np.where(np.abs(z_scores) > threshold)


# def outliers_iqr(ys):
#     quartile_1, quartile_3 = np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((ys > upper_bound) | (ys < lower_bound))


# # outlier_mask = outliers_z_score(data["selling_price"])
# outlier_mask1 = outliers_iqr(data["selling_price"])

data["selling_price"].value_counts()
outliers = data[
    ["selling_price "] > data["selling_price"].mean() + 3 * data["selling_price"].std()
]

pdb.set_trace()
