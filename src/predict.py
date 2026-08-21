from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from build_model import (PatchEmbedding,PositionEmbedding)



#LSTM model
def load_trained_model():

    model = load_model("models/LSTM_model.keras")

    return model


def predict(model, x_val):

    y_pred = model.predict(x_val)

    return y_pred


def plot_prediction(y_val, y_pred):
    plt.figure(figsize=(12,6))
    plt.plot(y_val, label="True RUL")
    plt.plot(y_pred, label="Predicted RUL")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/validation_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_model(y_val, y_pred):

    mae = mean_absolute_error(y_val,y_pred)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    r2 = r2_score(y_val, y_pred)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")

    return mae, rmse, r2


def save_predictions(y_val, y_pred):
    results = pd.DataFrame({"True_RUL": y_val, "Predicted_RUL": y_pred.flatten()})
    results.to_csv("outputs/validation_predictions.csv", index=False)
    print("Predictions saved successfully.")
                        
def plot_loss_val_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.savefig("figures/validation_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()



def worst_engine_predict(val_engine_ids, y_val, y_pred ):
    results = pd.DataFrame({"engine_id": val_engine_ids, "true_RUL": y_val, "pred_RUL": y_pred.flatten()})
    results["error"] = (results["true_RUL"] - results["pred_RUL"])
    results.to_csv("outputs/engine_ids_predictions.csv", index=False)
    return results

def err(y_test, y_predict):
    result = []
    for i in range(100):
        error_dict = {"engine": i+1}

        error_dict["Actual_y_test"] = y_test[i]

        error_dict["Predicted_y_test"] = y_predict[i] 

        error_dict["error"] = abs( y_test[i]- y_predict[i])
        if abs( y_test[i]- y_predict[i])>20:

              result.append(error_dict)
    errors = pd.DataFrame(result)
    errors.to_csv("outputs/error.csv", index=False)
    return errors   


#patchTST model
def load_trained_model_patchTST():

    model = load_model("models/patchTST_model.keras")

    return model

#def load_trained_model_patchTST():

    model = load_model(
        "models/patchTST_model.keras",
        custom_objects={
            "PatchEmbedding": PatchEmbedding,
            "PositionEmbedding": PositionEmbedding,
        }
    )

    return model


def predict(model, x_val):

    y_pred = model.predict(x_val)

    return y_pred


def plot_prediction(y_val, y_pred):
    plt.figure(figsize=(12,6))
    plt.plot(y_val, label="True RUL")
    plt.plot(y_pred, label="Predicted RUL")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/validation_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")

    return mae, rmse, r2


def save_predictions(y_val, y_pred):
    results = pd.DataFrame({"True_RUL": y_val, "Predicted_RUL": y_pred.flatten()})
    results.to_csv("outputs/validation_predictions.csv", index=False)
    print("Predictions saved successfully.")
                        
def prediction2(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.savefig("figures/validation_prediction.png", dpi=300, bbox_inches="tight")
    plt.show()
