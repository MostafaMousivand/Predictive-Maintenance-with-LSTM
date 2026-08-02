import pandas as pd
from data_loader import load_data
from preprocessing import (null_preprocess,max_min_difference_sensors,reminder_life,decline, rolling_mean, rolling_std)
from sequence_generator import (create_sequences, split, data_divided)
from build_model import (normalizes, create_model, earlier_stop, compiles, fits)
from predict import (load_trained_model, predict, plot_prediction,
evaluate_model, save_predictions, prediction2, worst_engine_predict)
important_sensor = ['sensor_3', 'sensor_4', 'sensor_9', 'sensor_11',
                     'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

#preprocessing
df = load_data()
df = null_preprocess(df)
range_df = max_min_difference_sensors(df)
slope_df = decline(df)
df["RUL"] = reminder_life(df)
df = df.merge(range_df, on="engine_id", how="left")
df = df.merge(slope_df, on="engine_id", how="left")
df = rolling_mean(df, important_sensor, 5)
df = rolling_std(df, important_sensor, 5)

train_engines, val_engines = split(df)
train_df, val_df = data_divided(df, train_engines, val_engines)
x_train, y_train, train_engine_ids = create_sequences(train_df, 30)
x_val, y_val, val_engine_ids = create_sequences(val_df, 30) 
#print(x_train.shape)
#print(x_val.shape)
x_train_sc, x_val_sc = normalizes(x_train, x_val)
lstm_model = create_model()
early_stop = earlier_stop()
compile = compiles(lstm_model)
history = fits(lstm_model, x_train_sc, y_train, x_val_sc, y_val, early_stop)
lstm_model.save("models/lstm_model.keras")

#prediction
model = load_trained_model()

y_pred = predict(model, x_val_sc)
print (y_pred)

plot_prediction(y_val, y_pred)

evaluate_model(y_val, y_pred)

save_predictions(y_val, y_pred)

prediction2(history)

worst_engine_predict(val_engine_ids, y_val, y_pred )

