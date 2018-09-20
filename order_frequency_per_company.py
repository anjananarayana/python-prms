import pandas as pd
import numpy as np
import sys

# opening the CSV with the right encoding
df = pd.read_csv("transactions.csv")
# df = pd.read_csv('transactions_test.csv')
# changing the data type of data time column
df["date_shipped"] = pd.to_datetime(df["date_shipped"], format="%Y-%m-%d")

# print number of rows and data type of each column of the dataframe
print(len(df))
print(df.dtypes)

# for each customer, finding its first order date and last order date, its most common order type, number of orders placed, mean/median between orders
company_df = df.groupby(["customer_num"]).agg(
    {
        "order_type": {"most_common_order_type": lambda x: x.value_counts().index[0]},
        "date_shipped": {
            "average_days_between_orders": lambda x: pd.Series(x.unique())
            .sort_values()
            .diff()
            .mean(),
            "median_days_between_orders": lambda x: pd.Series(x.unique())
            .sort_values()
            .diff()
            .median(),
            "max_gap_between_orders": lambda x: pd.Series(x.unique())
            .sort_values()
            .diff()
            .max(),
            "first_order_date": np.min,
            "last_order_date": np.max,
        },
        "order_num": {"num_orders": lambda x: len(x.unique())},
        "item_num": {"num_of_unique_items": lambda x: len(x.unique())},
    }
)


# finding the life time (last order date - first order date) of the company in days
company_df["company_lifetime_in_days"] = (
    company_df.date_shipped["last_order_date"]
    - company_df.date_shipped["first_order_date"]
)
print("Total Number of companies: {}".format(len(company_df)))

# removing all companies where lifetime is zero
company_df = company_df.loc[company_df["company_lifetime_in_days"].dt.days != 0]

# finding average number of orders per week
company_df["orders_per_week"] = (
    company_df.order_num["num_orders"] / company_df["company_lifetime_in_days"].dt.days
) * 7

# creating a churn column based on max_gap_between_orders column
churn_period_in_weeks = 24
company_df["Churned"] = np.where(
    company_df.date_shipped["max_gap_between_orders"].dt.days
    > (churn_period_in_weeks * 7),
    1,
    0,
)
## find average number of days between orders
# company_df['average_days_between_orders'] = company_df.date_shipped['sum_of_days_between_orders']/np.maximum(1,company_df.order_num['num_orders']-1)

# print(company_df.head())
print("Number of non zero companies: {}".format(len(company_df)))
print("Number of companies that churned: {}".format(company_df["Churned"].sum()))
print("Average # of orders per weeks: {}".format(company_df["orders_per_week"].mean()))
print("Std. dev of orders per weeks: {}".format(company_df["orders_per_week"].std()))

company_df.to_csv("ordering_frequency.csv")

