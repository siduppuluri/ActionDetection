import os
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from data_collection import actions, X_train, y_train, X_test, y_test


#BUILD AND TRAIN LSTM NEURAL NETWORK ####################################################


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# model.summary()


# MAKING PREDICTIONS ####################################################


res = model.predict(X_test)

actions[np.argmax(res[4])]
actions[np.argmax(res[y_test])]


# SAVE WEIGHTS ####################################################


model.save('action.keras')
model.load_weights('action.keras')


# CONFUSION MATRIX AND METRICS ####################################################


yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)

accuracy_score(ytrue, yhat)

