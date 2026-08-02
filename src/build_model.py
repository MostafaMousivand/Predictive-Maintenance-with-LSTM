from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential

def normalizes(x_train, x_val):
  x_train_new = x_train.reshape(-1, 40)
  x_val_new = x_val.reshape(-1, 40)
  sc = StandardScaler()
  x_train_sc = sc.fit_transform(x_train_new)
  x_val_sc = sc.transform(x_val_new)
  x_train_sc = x_train_sc.reshape(x_train.shape)
  x_val_sc = x_val_sc.reshape( x_val.shape)
  return x_train_sc, x_val_sc


def create_model():
    lstm_model = Sequential([
    LSTM(100 , return_sequences= True, dropout=0.2),
    LSTM(100, dropout= 0.2),
    Dense(1)])
    return lstm_model
    

def earlier_stop():    
    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True)
    return early_stop

def compiles(lstm_model):
   lstm_model.compile(optimizer= "adam", loss= "mean_squared_error", metrics = ["RootMeanSquaredError"])
   return lstm_model

def fits(lstm_model, x_train_sc, y_train, x_val_sc, y_val, early_stop):
    history = lstm_model.fit(x_train_sc, y_train, epochs= 25, validation_data = (x_val_sc, y_val), callbacks=[early_stop])
    return history