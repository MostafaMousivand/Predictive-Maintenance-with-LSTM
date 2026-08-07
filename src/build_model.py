import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import (Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import (MultiHeadAttention,LayerNormalization,Dropout,Conv1D,GlobalAveragePooling1D,Input,Dense)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import (Input,GlobalAveragePooling1D,Dense,Dropout)
from tensorflow.keras.optimizers import AdamW
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Layer, Dense, Reshape






#LSTM model
def create_model():
    lstm_model = Sequential([
    LSTM(100 , return_sequences= True, dropout=0.2),
    LSTM(100, dropout= 0.2),
    Dense(1)])
    return lstm_model

def compile_lstm(lstm_model):
    lstm_model.compile(optimizer= "adam", loss= "mean_squared_error", metrics = ["RootMeanSquaredError"])
    return lstm_model

def fit_lstm(lstm_model, x_train_sc, y_train, x_val_sc, y_val, early_stop):
    history = lstm_model.fit(x_train_sc, y_train, epochs= 25, validation_data = (x_val_sc, y_val), callbacks=[early_stop])
    return history


#transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def create_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.3)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def compile_transformer(model):
   optimizer = Adam(learning_rate=1e-4)
   model.compile(optimizer, loss= "mean_squared_error", metrics = ["RootMeanSquaredError"])
   return model

def fit_transformer(model, x_train_sc, y_train, x_val_sc, y_val, early_stop):
    history = model.fit(x_train_sc, y_train, epochs= 25, validation_data = (x_val_sc, y_val), callbacks=[early_stop])
    return history


#patchTST model
@register_keras_serializable()
class PatchEmbedding(Layer):

    def __init__( self, patch_length, num_features, embed_dim, **kwargs):

        super().__init__(**kwargs)

        self.patch_length = patch_length
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.flatten = Reshape((-1, patch_length * num_features))

        self.projection = Dense(embed_dim)

    def call(self, inputs):

        x = self.flatten(inputs)

        x = self.projection(x)

        return x

    def get_config(self):

        config = super().get_config()

        config.update({"patch_length": self.patch_length,"num_features": self.num_features,"embed_dim": self.embed_dim})

        return config
   
@register_keras_serializable()
class PositionEmbedding(Layer):

    def __init__(self,num_patches,embed_dim,**kwargs):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.position_embedding = self.add_weight(name="position_embedding",shape=(num_patches, embed_dim), initializer="random_normal",trainable=True)

    def call(self, inputs):

        return inputs + self.position_embedding

    def get_config(self):

        config = super().get_config()

        config.update({"num_patches": self.num_patches,"embed_dim": self.embed_dim})

        return config

def transformer_encoder(inputs,embed_dim,num_heads,ff_dim,dropout_rate=0.2):

    # -------- LayerNorm --------
    x = LayerNormalization(epsilon=1e-6)(inputs)

    # -------- Multi-Head Attention --------
    attention = MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim // num_heads,dropout=dropout_rate)(x,x)

    attention = Dropout(dropout_rate)(attention)

    # -------- Residual --------
    x = inputs + attention
    # -------- LayerNorm --------
    y = LayerNormalization(epsilon=1e-6)(x)
    # -------- Feed Forward --------
    y = Conv1D(filters=ff_dim,kernel_size=1, activation="gelu")(y)

    y = Dropout(dropout_rate)(y)

    y = Conv1D(filters=embed_dim,kernel_size=1)(y)

    # -------- Residual --------
    outputs = x + y

    return outputs

def create_patchTST(input_shape,embed_dim=128,num_heads=4,ff_dim=256,num_encoder=1,dropout_rate=0.2):

    inputs = Input(shape=input_shape)

    # Patch Embedding

    x = PatchEmbedding(patch_length=input_shape[1],num_features=input_shape[2],embed_dim=embed_dim)(inputs)

    # Position Embedding

    x = PositionEmbedding(num_patches=input_shape[0],embed_dim=embed_dim)(x)

    # Encoder block 1

    for _ in range(num_encoder):

       x = transformer_encoder(inputs=x,embed_dim=embed_dim,num_heads=num_heads,ff_dim=ff_dim,dropout_rate=dropout_rate)


    # جمع بندی Patchها
    x = GlobalAveragePooling1D()(x)

    # Regression Head

    x = Dense(64,activation="gelu")(x)

    x = Dropout(0.3)(x)

    outputs = Dense(1)(x)

    model = Model(inputs,outputs)

    return model

def compile_patchTST(model):

    optimizer = AdamW(learning_rate=1e-4,weight_decay=1e-4)

    model.compile(optimizer=optimizer,loss="mse",metrics=["mae", "RootMeanSquaredError"])

    return model

def patchTST_early_stop():

    return EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

def earlier_stop():    
    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True)
    return early_stop

