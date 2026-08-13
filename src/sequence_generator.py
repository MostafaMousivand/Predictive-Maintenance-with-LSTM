import numpy as np
from data_loader import load_data
from sklearn.model_selection import train_test_split


def split(df):

    engine_ids = df["engine_id"].unique()

    train_engines, val_engines = train_test_split(engine_ids,test_size=0.2,random_state=42)

    return train_engines, val_engines


def data_divided(df, train_engines, val_engines):

    train_df = df[df["engine_id"].isin(train_engines)]

    val_df = df[df["engine_id"].isin(val_engines)]

    return train_df, val_df

#For training data
def create_sequences(df, sequence_length):

    x = []
    y = []
    engine_ids = []

    for engine_id in df["engine_id"].unique():

        engine = df[df["engine_id"] == engine_id]

        features = engine.drop(columns=["engine_id", "cycle", "RUL"]).values

        target = engine["RUL"].values

        for i in range(len(engine) - sequence_length + 1):

            x.append(features[i:i+sequence_length])

            y.append(target[i+sequence_length-1])

            engine_ids.append(engine_id)

    return np.array(x), np.array(y), np.array(engine_ids)


#For testing data
def create_test_sequences(df, sequence_length):

    x = []
    engine_ids = []

    for engine_id in df["engine_id"].unique():

        engine = df[df["engine_id"] == engine_id]

        features = engine.drop(
            columns=["engine_id", "cycle"]
        ).values

        # تعداد سیکل‌های واقعی موتور
        n_cycles = len(features)

        # اگر موتور حداقل 75 سیکل دارد
        if n_cycles >= sequence_length:

            sequence = features[-sequence_length:]

        # اگر موتور کمتر از 75 سیکل دارد
        else:

            padding_size = sequence_length - n_cycles

            # اولین observation واقعی موتور
            first_observation = features[0:1]

            # تکرار اولین observation برای padding
            padding = np.repeat(first_observation,padding_size,axis=0)

            # padding در ابتدای sequence قرار می‌گیرد
            sequence = np.concatenate([padding, features],axis=0)

        x.append(sequence)
        engine_ids.append(engine_id)

    return np.array(x), np.array(engine_ids)




#patchTST model
import numpy as np


def create_patch_sequences(
        df,
        sequence_length,
        patch_length,
        stride):

    x = []
    y = []


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


        # ایجاد پنجره‌های زمانی
        for i in range(len(engine) - sequence_length + 1):
            sequence = features[i:i + sequence_length]

            rul = target[i + sequence_length - 1]

            # تبدیل sequence به patch
            patches = []

            for j in range(0,sequence_length - patch_length + 1,stride):

                patch = sequence[j:j + patch_length]

                patches.append(patch)

            x.append(patches)

            y.append(rul)


    return np.array(x), np.array(y)