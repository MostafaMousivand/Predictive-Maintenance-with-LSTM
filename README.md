# Predictive Maintenance and RUL Prediction with LSTM

A deep learning project for **predictive maintenance of turbofan engines** using the NASA C-MAPSS FD001 dataset. The objective is to estimate the **Remaining Useful Life (RUL)** of an engine from multivariate sensor time-series data.

## Project Overview

The project treats RUL estimation as a supervised time-series regression problem:

```text
Sensor measurements
        |
        v
Preprocessing
        |
        v
Feature Engineering
        |
        v
Time-Series Sequences
        |
        v
       LSTM
        |
        v
Predicted RUL
```

## Dataset

This project uses the **NASA C-MAPSS FD001** dataset.

FD001 contains run-to-failure trajectories for **100 turbofan engines**. Each observation contains an engine identifier, operating cycle, and multiple sensor measurements.

The RUL target is constructed as:

RUL = Maximum cycle of engine - Current cycle


## Data Preprocessing

The project includes:

- Missing-value handling
- Sensor analysis and selection
- Correlation analysis
- Engine-level degradation analysis
- Data splitting
- Feature normalization
- Sliding-window sequence generation

The main sensors selected for the final experiments were:

sensor_3, sensor_4, sensor_9, sensor_11,sensor_17



## Feature Engineering Experiments

Several approaches were tested:

### Trend-Based Features

Sensor range and degradation/trend features were investigated as additional information for the RUL model.

## Sequence Generation

Because RUL depends on the history of sensor measurements, the data is converted into sliding-window sequences.

The most effective sequence length found in the experiments was:

```text
sequence_length = 75
```

Example:

75 cycles of sensor history
            |
            v
           LSTM
            |
            v
        Predicted RUL


### Sequence Length Experiment

| Sequence Length |       MAE |       RMSE |        R² |
| --------------: | --------: | ---------: | --------: |
|              30 |    11.722 |     16.489 |     0.920 |
|          **75** | **7.680** | **10.700** | **0.949** |
|             100 |     8.250 |     10.903 |     0.931 |


The experiment showed that increasing the temporal context from 30 to 75 cycles produced a substantial improvement, while increasing it further to 100 cycles did not provide additional benefit.

## Final LSTM Model

The best-performing architecture was the original two-layer LSTM:

```python
def create_model():
    lstm_model = Sequential([
        LSTM(100, return_sequences=True, dropout=0.2),
        LSTM(100, dropout=0.2),
        Dense(1)
    ])
    return lstm_model
```

Architecture:

Input
  |
LSTM - 100 units
  |
Dropout - 0.2
  |
LSTM - 100 units
  |
Dropout - 0.2
  |
Dense - 1
  |
Predicted RUL


The model was compiled using the Adam optimizer and mean squared error loss.

Early stopping was used during training to retain the best validation model.

## Alternative Models

Several alternative architectures were investigated.

### Transformer

A Transformer encoder was implemented for multivariate time-series RUL prediction. Although the training process initially looked promising, validation loss increased after several epochs and the final performance did not outperform the LSTM baseline.

### PatchTST

PatchTST was also investigated using:

- Patch generation
- Patch embedding
- Positional embedding
- Transformer encoder blocks
- Regression head

Multiple combinations of embedding dimension, attention heads, feed-forward dimension, encoder depth, and dropout were tested.

The tested PatchTST configurations performed worse than the LSTM model on this project.

This experiment demonstrated that a more complex architecture does not automatically provide better performance for a particular dataset.

## Final Results

The best result obtained in the experiments was:

```text
MAE  : 7.680
RMSE : 10.700
R²   : 0.949
```

Configuration:

```text
Model          : 2-layer LSTM
LSTM units     : 100 + 100
Dropout        : 0.2
Sequence length: 75
Dataset        : C-MAPSS FD001
```

These metrics correspond to the validation/evaluation setup used in this project. They should not be interpreted as an official C-MAPSS benchmark result unless an independent test-set protocol is used.

## Project Structure


RUL_project/
|
├── data/
|
├── models/
|   └── lstm_model.keras
|
├── figures/
|
├── src/
|   ├── main.py
|   ├── data_loader.py
|   ├── preprocessing.py
|   ├── sequence_generator.py
|   ├── build_model.py
|   └── predict.py
|
├── requirements.txt
|
└── README.md


### Main Files

**`data_loader.py`**  
Loads and prepares the raw dataset.

**`preprocessing.py`**  
Contains preprocessing, sensor selection, feature engineering, splitting, and normalization functions.

**`sequence_generator.py`**  
Creates sliding-window sequences for the LSTM model.

**`build_model.py`**  
Contains model architectures, compilation, and training-related functions.

**`predict.py`**  
Loads trained models, generates predictions, calculates evaluation metrics, and creates prediction plots.

**`main.py`**  
Controls the complete workflow and allows the selected model/experiment to be executed.

## Installation

The project was developed with Python 3.10.

Install the main dependencies with:

pip install numpy pandas scikit-learn matplotlib seaborn scipy tensorflow


## Running the Project

From the project directory:


python src/main.py


## Evaluation Metrics

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual RUL. Lower is better.

### RMSE

Root Mean Squared Error gives greater weight to large prediction errors. Lower is better.

### R²

R² measures the proportion of target variance explained by the model. Higher is better.

## Key Findings

1. Sequence length had a major effect on RUL prediction performance.
2. A sequence length of 75 performed better than 30 and 100 in the tested configurations.
3. Rolling mean and rolling standard deviation did not provide a meaningful improvement.
4. PCA did not provide a meaningful improvement.
5. Transformer experiments did not outperform the final LSTM.
6. The tested PatchTST configurations performed worse than the LSTM.
7. The relatively simple two-layer LSTM achieved the best result obtained in this project.
8. The experiments show that appropriate temporal context and preprocessing can be more important than simply increasing model complexity.

## Future Work

Potential improvements include:

- Attention-based LSTM
- CNN-LSTM architectures
- Learning-rate optimization
- Huber loss and other robust loss functions
- More systematic hyperparameter optimization
- Additional degradation-oriented features
- Engine-specific normalization
- Evaluation on the official C-MAPSS test set
- Prediction uncertainty estimation
- Sensor importance and explainability
- Deployment as an industrial predictive-maintenance inference pipeline

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- SciPy

## Project Goal

The goal of this project is to develop a structured and reproducible deep-learning workflow for **industrial predictive maintenance and Remaining Useful Life estimation**.

The experiments also highlight an important practical lesson:

> A carefully tuned and appropriately sized model can outperform a more complex architecture when its structure matches the characteristics of the data.

## License

This project is intended for educational, portfolio, and research purposes. Please check the usage and redistribution terms of the original NASA C-MAPSS dataset before redistributing the dataset itself.
