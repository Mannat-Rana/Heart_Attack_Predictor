#Import numpy for np.unique()
import numpy as np

#Import pandas for data retrieval and storage
import pandas as pd

#Import tensorflow for DNN Modeling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

#Import sklearn functions for data splitting and processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

#Import matplotlib for plotting accuracy and loss
import matplotlib.pyplot as plt

#Use pandas to retrieve data from source
raw_data = pd.read_csv('https://raw.githubusercontent.com/Mannat-Rana/Heart_Attack_Predictor/main/heart.csv')

#Prepare data for DNN with sklearn functions
features = raw_data.drop(columns=['output'])
labels = raw_data['output'].values
normalizer = MinMaxScaler()
normalizer.fit(features)
features = normalizer.transform(features)
features, labels = shuffle(features, labels)
num_classes = len(np.unique(labels))
num_features = features.shape[1]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=1)

#Define DNN model
model = tf.keras.Sequential([tf.keras.layers.Input(shape = (n_features,)),
                             tf.keras.layers.Dense(32, activation='relu',kernel_initializer='he_normal'),
                             tf.keras.layers.Dense(16, activation='relu',kernel_initializer='he_normal'),
                             tf.keras.layers.Dropout(0.02), #Added to reduce overfitting
                             tf.keras.layers.Dense(32, activation='relu',kernel_initializer='he_normal'),
                             tf.keras.layers.Dense(1, activation='linear'),
                             tf.keras.layers.Dense(n_class, activation = 'softmax')
                             ])
#Compile Model
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(loss = ['mse', 'sparse_categorical_crossentropy'],
              metrics = [accuracy], 
              optimizer = 'adam')
#Verify Model is as intended
model.summary()

#Fit Model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=3, validation_data=(x_test, y_test))

#Gather and Plot Accuracy and Loss of the DNN
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc = 0)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc = 0)
plt.figure()
plt.show()
