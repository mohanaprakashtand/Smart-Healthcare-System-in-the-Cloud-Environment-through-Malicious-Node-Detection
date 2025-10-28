import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Classificaltion_Evaluation import ClassificationEvaluation


class RBMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=256, learning_rate=0.01, batch_size=10, n_iter=10, random_state=None):
        self.rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate, batch_size=batch_size,
                                n_iter=n_iter, random_state=random_state)

    def fit(self, X, y=None):
        self.rbm.fit(X, y)
        return self

    def transform(self, X):
        return self.rbm.transform(X)

    def fit_transform(self, X, y=None):
        return self.rbm.fit_transform(X, y)


def build_dbn(input_shape, hidden_units, output_units):
    model = Sequential()
    for units in hidden_units:
        model.add(Dense(units=units, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.2))
        input_shape = units
    model.add(Dense(units=output_units, activation='softmax'))
    return model


def Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target):
    # Define the RBM layers
    rbm1 = RBMTransformer(n_components=256, n_iter=10, random_state=42)
    rbm2 = RBMTransformer(n_components=128, n_iter=10, random_state=42)

    # Train the RBMs layer by layer and transform the data
    X_train_rbm1 = rbm1.fit_transform(Train_Data)
    X_train_rbm2 = rbm2.fit_transform(X_train_rbm1)

    X_test_rbm1 = rbm1.transform(Test_Data)
    X_test_rbm2 = rbm2.transform(X_test_rbm1)

    # Build the DBN model
    input_shape = X_train_rbm2.shape[1]
    hidden_units = [128, 64]  # Define the hidden layers of the DBN
    output_units = Train_Target.shape[1]
    dbn_model = build_dbn(input_shape, hidden_units, output_units)

    # Compile the DBN model
    dbn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the DBN model
    dbn_model.fit(X_train_rbm2, Train_Target, epochs=5, batch_size=4, validation_split=0.2)
    pred = dbn_model.predict(X_test_rbm2)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = ClassificationEvaluation(pred, Test_Target)
    return np.asarray(Eval), pred
