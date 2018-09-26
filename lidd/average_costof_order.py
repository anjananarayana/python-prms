import pandas as pd

data = pd.read_csv("/home/anjana/Desktop/short.csv", encoding="latin-1")
# finding the average cost of order
import pdb

# pdb.set_trace()
# average_data = data.groupby(["customer_num"]).agg({})
average_df = (
    data[["customer_num", "order_num", "item_num", "selling_price"]]
    .groupby(["customer_num", "order_num", "item_num"])
    .mean()
)


pdb.set_trace()


average_df_qua = (
    data[["customer_num", "order_num", "item_num", "qty_shipped"]]
    .groupby(["customer_num", "order_num"])
    .mean()
)
average_df.rename(columns={"selling_price": "average_order_cost"}, inplace=True)
average_df_qua.rename(columns={"qty_shipped": "average_quantity_order"}, inplace=True)

final_df = pd.merge(
    data, average_df, how="left", on=["customer_num", "order_num", "item_num"]
)


final_df_qua = pd.merge(
    final_df, average_df_qua, how="left", on=["customer_num", "order_num", "item_num"]
)
# import pdb

# pdb.set_trace()
final_df_qua.to_csv("cost_order.csv", index=False)
