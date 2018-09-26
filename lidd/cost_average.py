import pandas as pd
import numpy as np
import sys
import pdb

# opening the CSV with the right encoding
data = pd.read_csv("/home/anjana/anjana/csv/combined_transactions.csv")
# pdb.set_trace()
# caluculating the product between selling price and qty_shipped
data["average_total_sales"] = data["selling_price"] * data["qty_shipped"]
# print(data)
data1 = data.groupby(["customer_num", "order_num"]).sum()
# print(data1.head())
data2 = data1.reset_index()
data3 = data2[["customer_num", "average_total_sales"]].groupby(["customer_num"]).mean()
data4 = data3.reset_index()
# print(data4)
data4.to_csv("average_order_cost.csv", index=False)
df_qua = (
    data[["customer_num", "order_num", "qty_shipped"]]
    .groupby(["order_num", "customer_num"])
    .sum()
)
df_qua.rename(columns={"qty_shipped": "average_total_items"}, inplace=True)
data6 = df_qua.reset_index()
data7 = data6[["customer_num", "average_total_items"]].groupby(["customer_num"]).mean()
data7 = data7.reset_index()
data7.to_csv("average.csv", index=False)
final_df_qua = pd.merge(data7, data4, how="left", on=["customer_num"])
final_df_qua.to_csv("average_order_size.csv", index=False)

pdb.set_trace()
