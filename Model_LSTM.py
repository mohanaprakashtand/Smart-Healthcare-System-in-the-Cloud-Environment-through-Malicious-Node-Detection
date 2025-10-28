from keras import Sequential
import numpy as np
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_LSTM(trainX, trainY, testX, testy, BS=None, EP=None, sol=None):
    print('LSTM')

    if BS is None:
        BS = 32
    if EP is None:
        EP = 10
    if sol is None:
        sol = [10, 0.01]

    IMG_SIZE = [1, 100]
    num_classes = testy.shape[-1]

    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(Train_X.shape[1], Train_X.shape[-1])))
    model.add(Dense(int(sol[0]), activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, trainY, epochs=EP, steps_per_epoch=5, batch_size=BS, verbose=1,
              validation_data=(Test_X, testy))
    pred = model.predict(Test_X, verbose=2)
    pred = np.asarray(pred)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = ClassificationEvaluation(testy, pred)
    return Eval, pred
