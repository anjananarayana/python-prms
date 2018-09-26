import pandas as pd
import numpy as np
import sys

# opening the CSV with the right encoding
# df = pd.read_csv("/home/anjana/anjana/csv/combined_transactions.csv")
# df = pd.read_csv('transactions_test.csv')


import pandas as pd
import numpy as np

data = pd.read_csv("/home/anjana/Desktop/joy.csv")

data["total_rev"] = data["selling_price"] * data["qty_shipped"]
print(data)
data1 = data.groupby(["customer_num", "order_num"]).sum()
print(data1.head())
data2 = data1.reset_index()
data3 = data2[["customer_num", "total_rev"]].groupby(["customer_num"]).mean()
data4 = data3.reset_index()
print(data4)


# # print number of rows and data type of each column of the dataframe
# print(len(df))
# print(df.dtypes)
# df_qua = (
#     df[["customer_num", "order_num", "qty_shipped"]]
#     .groupby(["order_num", "customer_num"])
#     .sum()
# )
# import pdb


# df_qua.rename(columns={"qty_shipped": "average_order_size"}, inplace=True)
# df_qua_avg = pd.merge(df, df_qua, how="left", on=["customer_num", "order_num"])


# df_avg_qua = (
#     df_qua_avg[["customer_num", "average_order_size"]].groupby(["customer_num"]).mean()
# )

# final_df_qua = pd.merge(df, df_avg_qua, how="left", on=["customer_num"])

# import pdb


# final_df_qua[["customer_num", "average_order_size"]].to_csv(
#     "average_order_quality", index=False
# )
# pdb.set_trace()

