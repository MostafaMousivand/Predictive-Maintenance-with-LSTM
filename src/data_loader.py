import pandas as pd
import numpy as np

def load_data():

    columns = (
        ["engine_id", "cycle",
         "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv("data/train_FD001.txt", sep=r"\s+",header=None,names=columns)
    df = df.apply(pd.to_numeric, errors = "coerce")

    return df