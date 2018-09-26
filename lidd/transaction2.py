import pandas as pd
import numpy as np
import pdb
from datetime import date

data = pd.read_csv(
    "/home/anjana/Downloads/combined_transactions.csv", encoding="latin-1"
)
# avg_order_data = pd.read_csv(
#     "/home/anjana/anjana/PYTHON/chrun_prediction/average_order_size.csv"
# )
pdb.set_trace()
data["transaction_cost"] = data["selling_price"] * data["qty_shipped"]
# data["transaction_cost"] = data["selling_price"]
data["date_shipped"] = pd.to_datetime(data["date_shipped"], format="%Y-%m-%d")
group_by_monthyear = data.groupby(
    ["customer_num", data["date_shipped"].dt.strftime("%Y-%m")]
)
sum_monthyear = group_by_monthyear.sum()
sum_monthyear.tail(6)

top6_sum = sum_monthyear.groupby("customer_num").tail(6)
top6_sum = top6_sum.drop(["order_num", "selling_price"], axis=1)
top6_sum.rename(columns={"date_shipped": "per-month"}, inplace=True)
# top6_sum.rename(columns={"qty_shipped": "transaction_size"}, inplace=True)
top6_sum = top6_sum.reset_index()
top6_sum_pivot = top6_sum.pivot(
    index="customer_num", columns="date_shipped", values="transaction_cost"
)
months_below_august = list(
    filter(lambda x: "2018-02" < x < "2018-09", top6_sum_pivot.columns.tolist())
)

final_df = top6_sum_pivot[months_below_august]
final_df.fillna("0", inplace=True)
months = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}
for column in final_df.columns:
    year, month = column.split("-")
    final_df.rename(columns={column: year + "-" + months[month]}, inplace=True)

final_df = final_df.reset_index()

final_df.to_csv("trans_6mon_cost.csv", index=False)


top6_sum_pivot_qty = top6_sum.pivot(
    index="customer_num", columns="date_shipped", values="qty_shipped"
)
top6_sum_pivot_qty.rename(columns={"qty_shipped": "transaction_size"}, inplace=True)
final_df_qty = top6_sum_pivot_qty[months_below_august]
final_df_qty.fillna("0", inplace=True)
for column in final_df_qty.columns:
    year, month = column.split("-")
    final_df_qty.rename(columns={column: year + "-" + months[month]}, inplace=True)
final_df_qty.to_csv("trans_6mon_items.csv")
# final_df_size_cost = pd.merge(final_df, final_df_qty, how="left", on=["customer_num"])
# final_df_total = pd.merge(
#     final_df_size_cost, avg_order_data, how="left", on=["customer_num"]
# )
# final_df_total.to_csv("complete_transaction.csv", index=False)
pdb.set_trace()
