import pandas as pd
import numpy as np
import pdb

# data = pd.read_csv(
#     "/home/anjana/Downloads/combined_transactions.csv", encoding="latin-1"
# )
# std = data.std()
# mean = data.mean()
# # calculate the mean and sd for selling price get absolute value for the calculation
# data["outlier"] = (
#     abs(data["selling_price"] - mean["selling_price"]) > 2 * std["selling_price"]
# )
# data = data[["customer_num", "selling_price", "outlier"]]
# # converting the result(True / False) values into (1/0)
# data["outlier"] = data["outlier"].astype("category").cat.codes
# data.to_csv("outliers.csv", index=False)


data = pd.read_csv(
    "/home/anjana/Downloads/combined_transactions.csv",
    usecols=["selling_price", "customer_num"],
    encoding="latin-1",
)


std = data["selling_price"].std()
mean = data["selling_price"].mean()
# calculate the mean and sd for selling price get absolute value for the calculation
data["outlier"] = data["selling_price"].apply(
    lambda x: 1 if abs(x - mean) > 2 * std else 0
)
data.to_csv("outliers.csv", index=False)

# # pdb.set_trace()
# std = data["selling_price"].std()
# mean = data["selling_price"].mean()
# # calculate the mean and sd for selling price get absolute value for the calculation
# data["outlier"] = abs(data["selling_price"] - mean) > 2 * std
# data = data[["customer_num", "selling_price", "outlier"]]
# # converting the result(True / False) values into (1/0)
# # data["outlier"] = data["outlier"].astype("category").cat.codes
# # data.to_csv("outliers.csv", index=False)
# def f2(row):
#     if row["outlier"] == True:
#         val = 1
#     else:
#         val = 0
#     return val


# data["outlier"] = data.apply(f2, axis=1)


# print(data.head())
pdb.set_trace()

