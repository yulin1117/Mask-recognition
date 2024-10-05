import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2 as cv2
import os
from keras.callbacks import EarlyStopping, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Load the data
IMG_SIZE = 100
DATA_DIR = '/rhome/jimmy0111/jens/ml/observations/experiements/data'
CATEGORIES = ['with_mask', 'without_mask']

# Load and preprocess the data
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            print(e)

# Normalize the data
data = np.array(data) / 255
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels = np.array(labels)


#randomize the data
idx = np.random.permutation(len(data))
data, labels = data[idx], labels[idx]
plt.imshow(data[0])
plt.colorbar()
plt.savefig('/rhome/jimmy0111/jens/ml/observations/experiements/image.png')

# Split the data
split = int(len(data) * 0.6)
train_data = data[:split]
train_labels = labels[:split]
test_data = data[split:]
test_labels = labels[split:]

train_labels = tf.keras.utils.to_categorical(train_labels, 2)
test_labels = tf.keras.utils.to_categorical(test_labels, 2)

# Create the model
def model(IMG_SIZE):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(200, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(100, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    # x = Dropout(0.3)(x)
    outputs = Dense(2, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    plot_model(model, to_file='/rhome/jimmy0111/jens/ml/observations/experiements/model.png', show_shapes=True)
    return model

class LearningRateLogger(Callback):
    def on_train_begin(self, logs={}):
        self.lrates = []

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lrate = float(K.get_value(optimizer.lr))
        self.lrates.append(lrate)
        print('\nEpoch {} Lr {}'.format(epoch + 1, lrate))

    
def train(model, train_data, train_labels, test_data, test_labels):

    lr = 0.0005
    lr_decay = ExponentialDecay(lr, 30, 0.8,staircase=True)
    model.compile(optimizer=Adam(learning_rate=lr_decay), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # early_stopping = EarlyStopping(monitor='loss',min_delta=1e-2, patience=5)
    lr_logger = LearningRateLogger()

    history = model.fit(train_data, 
              train_labels, 
              epochs=50, 
              batch_size=50,
              validation_split=0.2,
              callbacks=[lr_logger])
    
    model.evaluate(test_data, test_labels, verbose=1)
    
    #plot loss
    image_path = '/rhome/jimmy0111/jens/ml/observations/experiements'
    plt.figure(figsize=(8,4.5),dpi = 200)
    plt.plot(history.history['loss'],'o-',markersize=2.5)
    plt.plot(history.history['val_loss'],'o-',markersize=2.5)   
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Train_loss', 'Test_val_loss'], fontsize=6)
    plt.savefig(image_path + '/loss.png')

    plt.figure(figsize=(8,4.5),dpi = 200)
    plt.plot(np.log10(history.history['loss']),'o-',markersize=2.5,label='log(loss)')
    plt.plot(np.log10(history.history['val_loss']),'o-',markersize=2.5,label='log(val_loss)')   
    plt.title('log(loss)')
    plt.xlabel('Epoch')
    plt.ylabel('log(loss)')
    plt.legend(['log(Train_loss)', '(Test_val_loss)'], fontsize=6)
    plt.savefig(image_path + '/log_loss.png')

    #plot accuracy
    plt.figure(figsize=(8,4.5),dpi = 200)
    plt.plot(history.history['accuracy'],'o-',markersize=2.5)
    plt.plot(history.history['val_accuracy'],'o-',markersize=2.5)
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train_accuracy', 'Test_val_accuracy'], fontsize=6)
    plt.savefig(image_path + '/accuracy.png')

    plt.figure(figsize=(8,4.5),dpi = 200)
    plt.plot(np.log10(history.history['accuracy']),'o-',markersize=2.5,label='log(accuracy)')
    plt.plot(np.log10(history.history['val_accuracy']),'o-',markersize=2.5,label='log(val_accuracy)')
    plt.title('log(accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('log(accuracy)')
    plt.legend(['log(Train_accuracy)', 'log(Test_val_accuracy)'], fontsize=6)
    plt.savefig(image_path + '/log_accuracy.png')

    # plot learning rate
    plt.figure(figsize=(8,4.5),dpi = 200)
    plt.plot(lr_logger.lrates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Time 1')
    plt.savefig(image_path + '/learning_rate.png')

train(model(IMG_SIZE), train_data, train_labels, test_data, test_labels)

#train refined model






