from keras.models import Sequential
from keras import optimizers, Input
from keras.layers import Dense, Dropout
from pandas import DataFrame

def build_model(X_train: DataFrame, activation_function: str) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32,activation=activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(32,activation=activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(64, activation=activation_function))
    model.add(Dense(1))
    
    ########################################################
    ########################################################
    opt = optimizers.Adam(learning_rate=0.001)
    
    model.compile(opt, loss ="mse", metrics =["accuracy"])
    return model
