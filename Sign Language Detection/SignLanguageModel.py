from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from SignLanguageUtils import SignLanguageUtils

sign_language_utils = SignLanguageUtils()

actions = sign_language_utils.actions
total_sequences = sign_language_utils.total_sequences
sequence_length = sign_language_utils.sequence_length
DATA_PATH = sign_language_utils.DATA_PATH

label_map = {label:num for num, label in enumerate(actions)}

videos, labels = [], []

for action in actions:
    for video in range(total_sequences):
        frames = []
        for frame_index in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(video), f"{frame_index}.npy"))
            frames.append(res)
        videos.append(frames)
        labels.append(label_map[action])

X = np.array(videos)
Y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=150)
model.save('sign_language_detector.h5')

# To test the model
result = model.predict(x_test)
for index in range(len(result)):
    print("PREDICTED VALUE:", np.argmax(result[index]))
    print("ACTUAL VALUE:", np.argmax(y_test[index]))