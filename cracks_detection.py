
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings

warnings.filterwarnings('ignore')

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Conv3D, MaxPooling3D, ReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

positive_1 = np.ones((1000, 1), dtype=int)
negative_0 = np.zeros((1000, 1), dtype=int)
row_positive_data = pd.read_csv('train_Positive_1_80-80.csv', header=None)
row_negative_data = pd.read_csv('train_Negative_1_80-80.csv', header=None)

row_data_1 = np.hstack([positive_1, row_positive_data])
row_data_0 = np.hstack([negative_0, row_negative_data])
data = np.vstack([row_data_0, row_data_1])
np.random.shuffle(data)
data_train = data[:1200]
data_val = data[1200:1600]
data_test = data[1600:]

data_train_x = data_train[:, 1:]
data_train_y = data_train[:, :1]
data_val_x = data_val[:, 1:]
data_val_y = data_val[:, :1]
data_test_x = data_test[:, 1:]
data_test_y = data_test[:, :1]

data_train_y, data_val_y, data_test_y = data_train_y.reshape(1200,), data_val_y.reshape(400,), data_test_y.reshape(400,)
data_train_x, data_val_x = data_train_x / 255, data_val_x / 255

data_train_x = data_train_x.reshape(1200, 80, 80, 1)
data_val_x = data_val_x.reshape(400, 80, 80, 1)
data_test_x = data_test_x.reshape(400, 80, 80, 1)

model = Sequential()

model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(80, 80, 1)))
model.add(MaxPool2D(4, 4))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(4, 4))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=RMSprop(lr=5e-5), loss='binary_crossentropy', metrics=['accuracy'])

data_train_y = to_categorical(data_train_y)
data_val_y = to_categorical(data_val_y)
data_test_y = to_categorical(data_test_y)

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

datagen.fit(data_train_x)

history = model.fit_generator(datagen.flow(data_train_x, data_train_y, batch_size=128),
                              epochs=50, validation_data=(data_val_x, data_val_y),
                              verbose=2, steps_per_epoch=data_train_x.shape[0] // 64)

test_loss, test_acc = model.evaluate(data_test_x, data_test_y)
results = [test_loss, test_acc]
df = pd.DataFrame(results, index=['test_loss', 'test_acc'])
predict = model.predict(data_test_x)
predict = pd.DataFrame(predict)
predict.to_csv('predict.csv')
df.to_csv('results.csv')

model.save('cracks_model.h5')

# loss_acc
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.savefig('loss_acc_graph.pdf')


'''
model = Sequential()

model.add(Conv2D(24, (20, 20), activation='relu', strides=2, input_shape=(256, 256, 3)))
model.add(MaxPool2D((7, 7), strides=2),)
model.add(Conv2D(48, (15, 15), activation='relu', strides=2))
model.add(MaxPool2D((4, 4), strides=2))
model.add(Conv2D(96, (10, 10), activation='relu', strides=2))
model.add(ReLU())
model.add(Conv2D(2, (1, 1), strides=1))
model.add(Dense(2, activation='softmax'))

model.summary()
'''
