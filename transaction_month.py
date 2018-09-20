import pandas as pd
import numpy as np
import pdb
from datetime import date

data = pd.read_csv(
    "/home/anjana/anjana/csv/combined_transactions.csv", encoding="latin-1"
)
data["transaction_cost"] = data["selling_price"] * data["qty_shipped"]
data["date_shipped"] = pd.to_datetime(data["date_shipped"], format="%Y-%m")
data["date_shipped"].rename()
group_by_monthyear = data.groupby(
    ["customer_num", data["date_shipped"].dt.strftime("%Y-%B")]
)
data["date_shipped"].rename(columns={"date_shipped": "per-month"}, inplace=True)
# grouping by sum of  the product b/w the selling_price & qty_shipped
sum_monthyear = group_by_monthyear.sum()
top6_sum = sum_monthyear.groupby("customer_num").head(6)
top6_sum = top6_sum.drop(["order_num", "selling_price"], axis=1)
top6_sum = top6_sum.reset_index()
top6_sum.rename(columns={"qty_shipped": "transaction_size"}, inplace=True)
top6_sum.rename(columns={"date_shipped": "per-month"}, inplace=True)
top6_sum.rename(columns={"transaction_size": "Total_items_purchased"}, inplace=True)
# save the data frame into transaction_size per month
top6_sum.to_csv("transaction_month.csv", index=False)
top6_sum_size = top6_sum.drop(["transaction_cost"], axis=1)
# save the data frame for  items _purchased per month
top6_sum_size.to_csv("Items_pur_month.csv", index=False)
pdb.set_trace()

