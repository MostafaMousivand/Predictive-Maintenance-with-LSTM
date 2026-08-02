import numpy as np
from data_loader import load_data
from sklearn.model_selection import train_test_split


def split(df):
    engine_ids = df["engine_id"].unique()

    train_engines, val_engines = train_test_split(
        engine_ids,
        test_size=0.2,
        random_state=42   
    )
    return train_engines, val_engines


def data_divided(df, train_engines, val_engines):
    train_df = df[df["engine_id"].isin(train_engines)]
    val_df = df[df["engine_id"].isin(val_engines)]
    return train_df, val_df


def create_sequences(df, sequence_length):

    x = []
    y = []
    engine_ids = []

    for engine_id in df["engine_id"].unique():

        engine = df[df["engine_id"] == engine_id]

        features = engine.drop(
            columns=[
                "engine_id",
                "cycle",
                "RUL"
            ]
        ).values

        target = engine["RUL"].values

        for i in range(
            len(engine) - sequence_length + 1
        ):

            x.append(
                features[i:i+sequence_length]
            )

            y.append(
                target[i+sequence_length-1]
            )
            engine_ids.append(
                engine_id
            )

    return np.array(x), np.array(y), np.array(engine_ids)


