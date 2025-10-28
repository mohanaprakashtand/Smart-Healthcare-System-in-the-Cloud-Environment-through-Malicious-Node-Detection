import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Multiply, Softmax, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


# =====================================================
# Multiplicative LSTM Layer (functional implementation)
# =====================================================
def multiplicative_lstm(inputs, units):
    """
    Implements multiplicative interaction before feeding to LSTM:
    m_t = (x_t * W_mx) ⊙ (x_t * W_mh)
    """
    input_proj = Dense(units, activation='tanh')(inputs)
    hidden_proj = Dense(units, activation='sigmoid')(inputs)
    mult_gate = Multiply()([input_proj, hidden_proj])
    lstm_out = LSTM(units, return_sequences=True)(mult_gate)
    return lstm_out


# =====================================================
# Attention Mechanism (Additive Attention)
# =====================================================
def attention_block(inputs):
    """
    inputs: (batch, time_steps, features)
    returns: context vector (batch, features)
    """
    # Compute attention scores
    score = Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention_weights = Dense(1, activation='softmax')(score)
    attention_weights = tf.nn.softmax(attention_weights, axis=1)

    # Weighted sum of input features
    context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
    return context_vector


# Adaptive Multiplicative LSTM + Attention Model
def Model_Ada_MLSTM_AM(trainX, trainY, testX, testy, BS=None, EP=None, sol=None):

    print("Adaptive Multiplicative LSTM with Attention (Ada-MLSTM-AM)")

    if BS is None:
        BS = 32
    if EP is None:
        EP = 10
    if sol is None:
        sol = [5, 5, 50]  # [dense_units, epoch, Step per epoch]

    IMG_SIZE = [1, 100]
    num_classes = testy.shape[-1]

    def reshape_data(X):
        Temp = np.zeros((X.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
        for i in range(X.shape[0]):
            Temp[i, :] = np.resize(X[i], (IMG_SIZE[0], IMG_SIZE[1]))
        return Temp.reshape(Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Train_X = reshape_data(trainX)
    Test_X = reshape_data(testX)

    inputs = Input(shape=(Train_X.shape[1], Train_X.shape[2]))

    # Multiplicative LSTM Core
    mlstm_output = multiplicative_lstm(inputs, 64)

    # Attention
    context_vector = attention_block(mlstm_output)

    # Adaptive scaling (learnable gate)
    adaptive_gate = Dense(64, activation='sigmoid')(context_vector)
    adaptive_output = Multiply()([context_vector, adaptive_gate])

    # Classification Head
    dense1 = Dense(sol[0], activation='relu')(adaptive_output)
    outputs = Dense(num_classes, activation='softmax')(dense1)

    # Compile Model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, trainY, epochs=int(sol[1]), batch_size=BS, steps_per_epoch=int(sol[2]),
              validation_data=(Test_X, testy), verbose=1)
    pred = model.predict(Test_X, verbose=2)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')

    Eval = ClassificationEvaluation(testy, pred)
    return Eval, pred
