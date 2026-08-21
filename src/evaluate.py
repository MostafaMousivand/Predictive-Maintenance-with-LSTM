from build_model import *
import pandas as pd
from data_loader import *
from data_loader import load_test_rul
from preprocessing import (null_preprocess,max_min_difference_sensors,reminder_life,decline,
                            health_index,rolling_mean, rolling_std, normalizes, normalize_patch)
from sequence_generator import (create_sequences, create_test_sequences, split, data_divided, create_patch_sequences)
from build_model import (create_model, earlier_stop, compile_lstm,
                        fit_lstm,compile_transformer, fit_transformer,
                          PatchEmbedding, PositionEmbedding, transformer_encoder, create_patchTST, compile_patchTST, patchTST_early_stop)
from predict import (load_trained_model,load_trained_model_patchTST, predict, plot_prediction,
evaluate_model,err, save_predictions, plot_loss_val_loss, worst_engine_predict)
important_sensor = ['sensor_3', 'sensor_4', 'sensor_9', 'sensor_11',
                   'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
from config import PATCHTST_EXPERIMENTS
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#preprocessing
df_test = load_test_data()

df_test = null_preprocess(df_test)

#range_df = max_min_difference_sensors(df_test)

#slope_df = decline(df_test)

#df["RUL"] = reminder_life(df)

#df_test = df_test.merge(range_df, on="engine_id", how="left")

#df_test = df_test.merge(slope_df, on="engine_id", how="left")

#df = rolling_mean(df, important_sensor, 5)
#df = rolling_std(df, important_sensor, 5)

print(df_test.shape)
print(df_test.columns)

#train_engines, val_engines = split(df)

#train_df, val_df = data_divided(df, train_engines, val_engines)

#x_train, y_train, train_engine_ids = create_sequences(df_test, 85)

#x_val, y_val, val_engine_ids = create_sequences(val_df, 85)

x_test, test_engine_ids = create_test_sequences(df_test, 120)
print(x_test.shape)

#df = health_index(df)

#x_train_sc, x_val_sc = normalizes(x_train, x_val)
#x_train_sc, x_val_sc, sc = normalizes(x_train, x_val, x_test )
sc = joblib.load("models/scaler.pkl")
""" print("Scaler features:", sc.n_features_in_)
print("Scaler mean:", sc.mean_)
print("Scaler scale:", sc.scale_) """

x_test_origin = x_test.reshape(-1, 9)
x_test_sc = sc.transform(x_test_origin)
x_test_sc = x_test_sc.reshape(x_test.shape)
model = load_trained_model()

""" print("Scaled Test mean:", x_test_sc.reshape(-1,19).mean(axis=0))
print("Scaled Test std:", x_test_sc.reshape(-1,19).std(axis=0))
for name, mean, std in zip(df_test.drop(columns=["engine_id","cycle"]).columns, x_test_sc.reshape(-1,19).mean(axis=0), x_test_sc.reshape(-1,19).std(axis=0)): print(name, "mean=", round(mean,3), "std=", round(std,3)) """


y_predict = model.predict(x_test_sc)
y_predict = y_predict.flatten()
y_test = load_test_rul()
print(y_predict.shape)

mae = mean_absolute_error(y_test, y_predict)

rmse = np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.3f}")


errors = np.abs(y_test - y_predict)
print("Mean error:", errors.mean())
print("Median error:", np.median(errors))
print("Max error:", errors.max())
errors = err(y_test, y_predict)
    

    #print(f"Engine {i+1}: Actual={y_test[i]:.1f}, Predicted={y_predict[i]:.1f}, Error={errors[i]:.1f}")

low = y_test <= 30
medium = (y_test > 30) & (y_test <= 80)
high = y_test > 80
print("Low RUL MAE:", mean_absolute_error(y_test[low], y_predict[low]))
print("Medium RUL MAE:", mean_absolute_error(y_test[medium], y_predict[medium]))
print("High RUL MAE:", mean_absolute_error(y_test[high], y_predict[high]))