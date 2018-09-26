import pandas as pd
import numpy as np
import pdb
from datetime import date
import calendar

data = pd.read_csv(
    "/home/anjana/Downloads/combined_transactions.csv", encoding="latin-1"
)

data4 = pd.read_csv(
    "/home/anjana/anjana/PYTHON/chrun_prediction/average_order_size.csv"
)
# data = pd.read_csv("/home/anjana/Desktop/short.csv", encoding="latin-1")
data["transaction_cost"] = data["selling_price"] * data["qty_shipped"]
data["date_shipped"] = pd.to_datetime(data["date_shipped"], format="%Y-%m")
data["date_shipped"].rename()

group_by_monthyear = data.groupby(
    ["customer_num", data["date_shipped"].dt.strftime("%Y -%B")]
)
data["date_shipped"].rename(columns={"date_shipped": "per-month"}, inplace=True)
# grouping by sum of  the product b/w the selling_price & qty_shipped
sum_monthyear = group_by_monthyear.sum()
top6_sum = sum_monthyear.groupby("customer_num").tail(7)
top6_sum = top6_sum.drop(["order_num", "selling_price"], axis=1)
top6_sum = top6_sum.reset_index()
top6_sum.rename(columns={"qty_shipped": "Total_items_purchased"}, inplace=True)
top6_sum.rename(columns={"date_shipped": "per-month"}, inplace=True)
# save the data frame into transaction_size per month
top6_sum5 = top6_sum[["customer_num", "per-month", "transaction_cost"]]
# pivoting the rows into columns
top6_sum5 = top6_sum.pivot(
    index="customer_num", columns="per-month", values="transaction_cost"
)
top6_sum5 = top6_sum5.reset_index()


# top6_sum5 = top6_sum5.fillna(0)
top6_sum5.fillna("0", inplace=True)
final_df = pd.merge(top6_sum5, data4, how="left", on=["customer_num"])

final_df.to_csv("transaction_cost_month_6.csv", index=False)

top6_sum_size = top6_sum.drop(["transaction_cost"], axis=1)
# save the data frame for  items _purchased per month
df = top6_sum_size.pivot(
    index="customer_num", columns="per-month", values="Total_items_purchased"
)
top6_sum_size = df.reset_index()
top6_sum_size.fillna("0", inplace=True)
final_df2 = pd.merge(top6_sum_size, data4, how="left", on=["customer_num"])
final_df2.to_csv("Items_per_month_6.csv", index=False)

