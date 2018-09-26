import pandas as pd

# opening the CSV with the right encoding
df = pd.read_csv("Sample Order file.csv", encoding="iso-8859-1")

# print number of rows and data type of each column
# print(len(df))
# print(df.dtypes)

# replacing the '.' in cost column with 0
df = df.replace(["."], ["0"])
# changing datatype of cost column to float
df[["cost"]] = df[["cost"]].astype(float)

# print number of rows and data type of each column
# print(len(df))
# print(df.dtypes)

# for each item finding its average sale and cost price
items_info = df.groupby(["item_num", "desc"]).agg(
    {"selling_price": "mean", "cost": "mean"}
)

print(items_info.head())

print("Number of unique items: {}".format(len(df.item_num.unique())))
print("Number of customers: {}".format(len(df.customer_num.unique())))
# writing the items information using the same encoding
items_info.to_csv("items_info.csv", encoding="iso-8859-1")

