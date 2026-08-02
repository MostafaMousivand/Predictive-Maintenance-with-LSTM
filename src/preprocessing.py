import pandas as pd
import numpy as np
from scipy.stats import linregress



def null_preprocess(df):
     df.fillna(df.mean(numeric_only = True), inplace= True)
     return df



def max_min_difference_sensors(df):

    important_sensor = [
        "sensor_3",
        "sensor_4",
        "sensor_9",
        "sensor_11",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21"
    ]

    results = []

    for i in range(1, 101):

        engine = df[df["engine_id"] == i]

        row = {"engine_id": i}

        for sensor in important_sensor:

            sensor_difference = round(
                engine[sensor].max() - engine[sensor].min(),
                2
            )

            row[f"{sensor}_range"] = sensor_difference

        results.append(row)

    difference = pd.DataFrame(results)

    return difference
    


def correlation(df):
    results = {}

    for i in range(1, 101):
        engine = df[df["engine_id"] == i]

        correlations = (
            engine.corr(numeric_only=True)["cycle"]
                  .sort_values(ascending=False)
        )

        results[i] = (correlations)

    return results


def reminder_life(df):
    reminder = []    
    for i in range(1, 101):
        engine = df[df["engine_id"] == i] 
        RUL = engine["cycle"].max() - engine["cycle"]
        reminder.extend(RUL)
        
    return reminder



def decline(df):

    sensor_cols = [
        "sensor_3","sensor_4","sensor_9","sensor_11",
        "sensor_15","sensor_17","sensor_20","sensor_21"
    ]

    results = []

    for i in range(1, 101):

        engine = df[df["engine_id"] == i]

        row = {"engine_id": i}

        for sensor in sensor_cols:

            slope = linregress(
                engine["cycle"],
                engine[sensor]
            ).slope

            row[f"{sensor}_slope"] = round(slope, 4)

        results.append(row)

    slope_df = pd.DataFrame(results)
    return slope_df



def rolling_mean(df, sensors, window):

    df = df.copy()

    for sensor in sensors:

        df[f"{sensor}_rolling_mean"] = np.nan

        for engine_id in df["engine_id"].unique():

            engine = df[df["engine_id"] == engine_id]

            rolling_values = []

            for i in range(len(engine)):

                start = max(0, i - window + 1)

                values = engine[sensor].iloc[start:i+1]

                rolling_values.append(values.mean())

            df.loc[engine.index, f"{sensor}_rolling_mean"] = rolling_values

    return df


import numpy as np

def rolling_std(df, sensor, window=5):

    df = df.copy()

    for sensor in sensor:

        rolling_values = []

        for engine_id in df["engine_id"].unique():

            engine = df[df["engine_id"] == engine_id]

            values = engine[sensor].tolist()

            for i in range(len(values)):

                start = max(0, i - window + 1)

                window_values = values[start:i+1]

                std = round(np.std(window_values), 2)

                rolling_values.append(std)

        df[f"{sensor}_rollstd_{window}"] = rolling_values

    return df