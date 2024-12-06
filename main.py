# Importar las bibliotecas necesarias:
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

path_train = 'archive (2)/train'
path_train = 'archive (2)/test'


# Preparar el conjunto de datos:
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='validation')

#Construir el modelo:

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))  # 7 clases

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Configurar los callbacks:
checkpoint = ModelCheckpoint('modelo_emociones.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')


#Entrenar el modelo:
epochs = 50

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs,
    #callbacks=[checkpoint, early_stopping]
    callbacks=[checkpoint]
)








