import pandas as pd
import numpy as np
import pdb

data = pd.read_csv(
    "/home/anjana/anjana/PYTHON/chrun_prediction/trans_4mon_cost.csv",
    encoding="latin-1",
)
# data = pd.read_csv(
#     "/home/anjana/anjana/PYTHON/chrun_prediction/trans_4mon_items.csv",
#     encoding="latin-1",
# )


def f(row):
    if row["2018-May"] == 0 and row["2018-June"] == 0:
        val = -100
    elif row["2018-May"] == row["2018-June"]:
        val = 0
    elif row["2018-May"] > row["2018-June"]:
        val = -1
    else:
        val = 1
    return val


def f1(row):
    if row["2018-June"] == 0 and row["2018-July"] == 0:
        val = -100
    elif row["2018-June"] == row["2018-July"]:
        val = 0
    elif row["2018-June"] > row["2018-July"]:
        val = -1
    else:
        val = 1
    return val


def f2(row):
    if row["2018-July"] == 0 and row["2018-August"] == 0:
        val = -100
    elif row["2018-July"] == row["2018-August"]:
        val = 0
    elif row["2018-July"] > row["2018-August"]:
        val = -1
    else:
        val = 1
    return val


data["May_June(%)"] = data.apply(f, axis=1)
data["June_July(%)"] = data.apply(f1, axis=1)
data["July_Aug(%)"] = data.apply(f2, axis=1)


# # data_dff_apr = data["2018-April"] - data["2018-March"]
# # data_dff_may = data["2018-May"] - data["2018-April"]
# data_dff_june = data["2018-June"] - data["2018-May"]
# data_dff_july = data["2018-July"] - data["2018-June"]
# data_dff_aug = data["2018-August"] - data["2018-July"]


# # data["percent_April(%)"] = (data_dff_apr / data["2018-April"]) * 100
# # data["percent_May(%)"] = (data_dff_may / data["2018-May"]) * 100
# data["May_June(%)"] = (data_dff_june / data["2018-May"]) * 100
# data["June_July(%)"] = (data_dff_july / data["2018-June"]) * 100
# data["July_Aug(%)"] = (data_dff_aug / data["2018-July"]) * 100
# # data["percent_March(%)"] = round(data["percent_March(%)"], 2)
# # data["percent_April(%)"] = round(data["percent_April(%)"], 2)
# # data["percent_May(%)"] = round(data["percent_May(%)"], 2)
# data["May_June(%)"] = round(data["May_June(%)"], 2)
# data["June_July(%)"] = round(data["June_July(%)"], 1)
# data["July_Aug(%)"] = round(data["July_Aug(%)"], 1)


data.to_csv("revenue_transaction_cost_4.csv", index=False)
pdb.set_trace()
