import pandas as pd
import numpy as np

def load_data():
    unimportant_sensor = ["setting_1", "setting_2", "setting_3","sensor_1","sensor_5","sensor_6","sensor_8",
                          "sensor_10","sensor_13", "sensor_15","sensor_16","sensor_18","sensor_19","sensor_20","sensor_21"]
    
    columns = (["engine_id", "cycle","setting_1", "setting_2", "setting_3"]+ [f"sensor_{i}" for i in range(1, 22)])

    df = pd.read_csv("data/train_FD001.txt", sep=r"\s+",header=None,names=columns)
    df = df.apply(pd.to_numeric, errors = "coerce")
    df = df.drop(columns = unimportant_sensor)

    return df

def load_test_data():
    unimportant_sensor = ["setting_1", "setting_2", "setting_3","sensor_1","sensor_5","sensor_6","sensor_8",
                              "sensor_10","sensor_13", "sensor_15","sensor_16","sensor_18","sensor_19","sensor_20","sensor_21"]
        
    columns = (["engine_id", "cycle","setting_1", "setting_2", "setting_3"]+ [f"sensor_{i}" for i in range(1, 22)])
    
    df_test = pd.read_csv("data/test_FD001.txt", sep=r"\s+",header=None,names=columns)
    df_test = df_test.apply(pd.to_numeric, errors = "coerce")
    df_test = df_test.drop(columns = unimportant_sensor)

    return df_test

def load_test_rul():
    rul = pd.read_csv("data/RUL_FD001.txt", header=None)
    return rul.iloc[:,0].values