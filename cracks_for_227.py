import os
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator

# copy the images to path


original_dataset_positive_dir = '/data-tmp/cracks/datasets/row_data/Positive'
original_dataset_negative_dir = '/data-tmp/cracks/datasets/row_data/Negative'
# original_dataset_dir = '/data-tmp/cracks/datasets/processed_data'

'''
base_dir = '/data-tmp/cracks/model'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_positive_dir = os.path.join(train_dir, 'positive')
os.mkdir(train_positive_dir)

train_negative_dir = os.path.join(train_dir, 'negative')
os.mkdir(train_negative_dir)

validation_positive_dir = os.path.join(validation_dir, 'positive')
os.mkdir(validation_positive_dir)

validation_negative_dir = os.path.join(validation_dir, 'negative')
os.mkdir(validation_negative_dir)

test_positive_dir = os.path.join(test_dir, 'positive')
os.mkdir(test_positive_dir)

test_negative_dir = os.path.join(test_dir, 'negative')
os.mkdir(test_negative_dir)

result_graph_dir = os.path.join(base_dir, 'result_graph')
os.mkdir(result_graph_dir)

result_model_dir = os.path.join(base_dir, 'result_model')
os.mkdir(result_model_dir)

result_version_dir = os.path.join(base_dir, 'result_version')
os.mkdir(result_version_dir)
'''


train_positive_dir = '/data-tmp/cracks/model/train/positive'
validation_positive_dir = '/data-tmp/cracks/model/validation/positive'
validation_negative_dir = '/data-tmp/cracks/model/validation/negative'
train_negative_dir = '/data-tmp/cracks/model/train/negative'
test_positive_dir = '/data-tmp/cracks/model/test/positive'
test_negative_dir = '/data-tmp/cracks/model/test/negative'
test_dir = '/data-tmp/cracks/model/test'


# processing copy
def copy_images(from_path, for_path, range_1, range_2):
    if range_1 == 1:
        fnames = ['{}.jpg'.format(str(i).zfill(5)) for i in range(1, range_2)]
    else:
        fnames = ['{}.jpg'.format(str(i).zfill(5)) for i in range(range_1, range_2)]
    for fname in fnames:
        src = os.path.join(from_path, fname)
        dst = os.path.join(for_path, fname)
        shutil.copy(src, dst)
    return 0


# the number of images
print('total training positive images:', len(os.listdir(train_positive_dir)))

# the dir of cracks
dir = {'train_positive': [original_dataset_positive_dir, train_positive_dir, 1, 1600],
       'validation_positive': [original_dataset_positive_dir, validation_positive_dir, 1600, 1800],
       'test_positive': [original_dataset_positive_dir, test_positive_dir, 1800, 2000],
       'train_negative': [original_dataset_negative_dir, train_negative_dir, 1, 1600],
       'validation_negative': [original_dataset_negative_dir, validation_negative_dir, 1600, 1800],
       'test_negative': [original_dataset_negative_dir, test_negative_dir, 1800, 2000]}


def extract_keys(keywords):
    from_path, for_path, range_1, range_2 = dir['{}'.format(keywords)]
    return from_path, for_path, type, range_1, range_2


for word in dir:
    from_path, for_path, type, range_1, range_2 = extract_keys(word)
    range_1, range_2 = int(range_1), int(range_2)
    copy_images(from_path, for_path, range_1, range_2)


# built model

model = Sequential()

model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(227, 227, 3)))
model.add(MaxPool2D(4, 4))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(4, 4))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# set the path
train_dir = '/data-tmp/cracks/model/train'
validation_dir = '/data-tmp/cracks/model/validation'

# processing the data
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=40,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(227, 227),
    batch_size=40,
    class_mode='binary'
)

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=40,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=20)

# model.save('/data-tmp/cracks/model/result_model/cracks_for_227.hs')


def loss_and_acc_graph(history):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    plt.show()
    # plt.savefig('loss_acc_graph.pdf')


loss_and_acc_graph(history)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=20,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)





