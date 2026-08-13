from build_model import *
import pandas as pd
from data_loader import load_data
from preprocessing import (null_preprocess,max_min_difference_sensors,reminder_life,decline,
                            health_index,rolling_mean, rolling_std, normalizes,normalize_patch)
from sequence_generator import (create_sequences, split, data_divided, create_patch_sequences)
from build_model import (create_model, earlier_stop, compile_lstm,
                        fit_lstm,compile_transformer, fit_transformer,
                          PatchEmbedding, PositionEmbedding, transformer_encoder, create_patchTST, compile_patchTST, patchTST_early_stop)
from predict import (load_trained_model,load_trained_model_patchTST, predict, plot_prediction,
evaluate_model, save_predictions, prediction2, worst_engine_predict)
important_sensor = ['sensor_3', 'sensor_4', 'sensor_9', 'sensor_11',
                   'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
from config import PATCHTST_EXPERIMENTS
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#preprocessing
df = load_data()

df = null_preprocess(df)

#ange_df = max_min_difference_sensors(df)

#slope_df = decline(df)

df["RUL"] = reminder_life(df)

#df = df.merge(range_df, on="engine_id", how="left")

#df = df.merge(slope_df, on="engine_id", how="left")

#df = rolling_mean(df, important_sensor, 5)
#df = rolling_std(df, important_sensor, 5)

print(df.shape)

train_engines, val_engines = split(df)

train_df, val_df = data_divided(df, train_engines, val_engines)

x_train, y_train, train_engine_ids = create_sequences(train_df, 120)
print(x_train.shape)
x_val, y_val, val_engine_ids = create_sequences(val_df, 120)

#df = health_index(df)

x_train_sc, x_val_sc, sc = normalizes(x_train, x_val)
joblib.dump(sc, "models/scaler.pkl")

#print("Train feature mean:", x_train.reshape(-1,19).mean(axis=0))
#print("Train feature std:", x_train.reshape(-1,19).std(axis=0))
#print(df.drop(columns=["engine_id","cycle","RUL"]).columns.tolist())


#create LSTM model1
lstm_model = create_model()
early_stop = earlier_stop()
lstm_model = compile_lstm(lstm_model)
history = fit_lstm(lstm_model, x_train_sc, y_train, x_val_sc, y_val, early_stop)
lstm_model.save("models/lstm_model.keras")


#create LSTM model github
# lstm_model = create_lstm_github(input_shape=x_train_sc.shape[1:])
# early_stop = earlier_stop()
# lstm_model = compile_lstm_github(lstm_model)
# history = fit_lstm_github(lstm_model, x_train_sc, y_train, x_val_sc, y_val, early_stop)
# lstm_model.save("models/lstm_github_model.keras")





#create transformer model
#transformer_model = create_transformer(input_shape=x_train_sc.shape[1:])
#transformer_model = compile_transformer(transformer_model)
#early_stop = earlier_stop()
#history = fit_transformer(transformer_model, x_train_sc, y_train, x_val_sc, y_val, early_stop)
#transformer_model.save("models/transformer_model.keras")

#LSTM prediction 
model = load_trained_model()

y_pred = predict(lstm_model, x_val_sc)

print (y_pred)

plot_prediction(y_val, y_pred)

evaluate_model(y_val, y_pred)

save_predictions(y_val, y_pred)

prediction2(history)

worst_engine_predict(val_engine_ids, y_val, y_pred )


#create patchTST model
# x_train, y_train = create_patch_sequences(train_df,sequence_length=30,patch_length=3,stride=5)

# x_val, y_val = create_patch_sequences(val_df,sequence_length=30,patch_length=3,stride=5)

# x_train_sc,x_val_sc = normalize_patch(x_train, x_val)

# results = []
# for exp in PATCHTST_EXPERIMENTS:
#     print("=" * 60)
#     print(f"Running {exp['name']}")
#     print("=" * 60)

#     model = create_patchTST(
#         input_shape=x_train.shape[1:],embed_dim=exp["embed_dim"],num_heads=exp["num_heads"],
#         ff_dim=exp["ff_dim"],num_encoder=exp["num_encoder"],dropout_rate=exp["dropout"])
    
#     model=compile_patchTST(model)

#     early_stop=patchTST_early_stop()

#     history = model.fit(x_train_sc,y_train,validation_data=(x_val_sc,y_val),
#                        epochs=100,batch_size=64,callbacks=[early_stop])
#     print("Training Finished")
#     y_pred=model.predict(x_val_sc,verbose=0)
#     mae=mean_absolute_error(y_val,y_pred)
#     rmse=np.sqrt(mean_squared_error(y_val,y_pred))
#     r2=r2_score(y_val,y_pred)
#     print(f"MAE  : {mae:.3f}")
#     print(f"RMSE : {rmse:.3f}")
#     print(f"R2   : {r2:.3f}")
#     print("-"*50)
#     results.append({"Experiment":exp["name"],"Embed":exp["embed_dim"],"Heads":exp["num_heads"],
#                     "FF":exp["ff_dim"],"Encoder":exp["num_encoder"],"Dropout":exp["dropout"],"MAE":mae,"RMSE":rmse,"R2":r2})
    
#     model.save("models/patchTST_model.keras")
# results_df=pd.DataFrame(results)
# results_df=results_df.sort_values(by="RMSE")
# print(results_df)
# results_df.to_csv("results/patchTST_results.csv",index=False)
# best=results_df.iloc[0]
# print("\nBest Experiment")
# print(best)
# print(df)

# patch_embedding = PatchEmbedding(patch_length=3,num_features=40,embed_dim=64)

# x_embed = patch_embedding(x_train_sc)

# position_embedding = PositionEmbedding(num_patches=6,embed_dim=64)

# x_position = position_embedding(x_embed)

# encoder_output = transformer_encoder(inputs=x_position,embed_dim=64,num_heads=4,ff_dim=128,dropout_rate=0.2)





#patchTST prediction 
# model = load_trained_model_patchTST()
# y_pred = predict(model, x_val_sc)
# print (y_pred)

# plot_prediction(y_val, y_pred)

# evaluate_model(y_val, y_pred)

# save_predictions(y_val, y_pred)

# prediction2(history)

#worst_engine_predict(val_engine_ids, y_val, y_pred )
