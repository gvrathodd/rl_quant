import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_regimes(prices, h):
    n = len(prices)
    regimes = np.full(n, np.nan)  # initialize with NaN for full length

    for i in range(n - h):
        window = prices.iloc[i:i + h + 1]
        ret_fwd = (window.iloc[-1] - window.iloc[0]) / window.iloc[0] * 100

        # numeric encoding: bull=2, wbull=1, sideways=0, wbear=-1, bear=-2
        if ret_fwd > 0.1:
            regime = 2
        elif 0.02 < ret_fwd <= 0.1:
            regime = 1
        elif -0.02 <= ret_fwd <= 0.02:
            regime = 0
        elif -0.1 <= ret_fwd < -0.02:
            regime = -1
        else:
            regime = -2

        regimes[i] = regime

    # fill the remaining trailing NaNs (for unmatched windows) with 0
    regimes = np.nan_to_num(regimes, nan=0)
    return regimes

h_values = list(range(50, 2001, 50))

for d in range(229):
    data = pd.read_csv(f"../../EBY/day{d}.csv")
    data_pb = data.iloc[:, 1:218]
    data_pb = data_pb.drop(data_pb.columns[71], axis=1)
    data_pb = data_pb.drop(data_pb.columns[71], axis=1)
    data_pb = data_pb.drop(data_pb.columns[24], axis=1).dropna()

    prices = data_pb["Price"]
    n = len(prices)

    # compute multiple label columns only
    label_df = pd.DataFrame(index=range(n))

    for h in h_values:
        col_name = f"class_h{h}"
        label_df[col_name] = compute_regimes(prices, h)

    # save only label columns
    label_df.to_csv(f"../data/label1/day{d}.csv", index=False)
