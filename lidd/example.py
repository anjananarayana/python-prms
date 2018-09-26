import pandas as pd
import numpy as np
import pdb
from datetime import date
import calendar

data = pd.read_csv(
    "/home/anjana/Downloads/combined_transactions.csv", encoding="latin-1"
)
# data = pd.read_csv("/home/anjana/Desktop/short.csv")


# # def outliers_z_score(ys):
# #     threshold = 3

# #     mean_y = np.mean(ys)
# #     stdev_y = np.std(ys)
# #     z_scores = [(y - mean_y) / stdev_y for y din ys]
# #     return np.where(np.abs(z_scores) > threshold)


# def outliers_iqr(ys):
#     quartile_1, quartile_3 = np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((ys > upper_bound) | (ys < lower_bound))


# # # # outlier_mask = outliers_z_score(data["selling_price"])
# # outlier_mask1 = outliers_iqr(data["selling_price"])

# # outlier_mask1.value_counts()

# # pdb.set_trace()


# std = data.std()
# mean = data.mean()
# data["outlier"] = (
#     abs(data["selling_price"] - mean["selling_price"]) > 2 * std["selling_price"]
# )
# data = data[["customer_num", "outlier", "selling_price"]]

# data["outlier"] = data["outlier"].astype("category").cat.codes


# datelisttemp = pd.date_range("1/1/2014", periods=3, freq="D")
# s = list(datelisttemp) * 3
# print(s.sort())


# # data = pd.read_csv("/home/anjana/Desktop/short.csv", encoding="latin-1")
# data["date_shipped"] = pd.to_datetime(data["date_shipped"], format="%Y-%m")

# # group_by_monthyear = data.groupby(
# #     ["customer_num", "order_num", data["date_shipped"].dt.strftime("%Y - %B")]
# # )

df_qua_max = data[["customer_num", "date_shipped"]].groupby(["customer_num"]).max()
data_max = df_qua_max.reset_index()

df_qua_min = data[["customer_num", "date_shipped"]].groupby(["customer_num"]).min()
data_min = df_qua_min.reset_index()
order_date = pd.merge(df_qua_max, df_qua_min, how="left", on=["customer_num"])
order_date = order_date.reset_index()

order_date.to_csv("fir_last_order_1.csv", index=False)


data["year"] = pd.DatetimeIndex(data["date_shipped"]).year
data["month"] = pd.DatetimeIndex(data["date_shipped"]).month

pdb.set_trace()
