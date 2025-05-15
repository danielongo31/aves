import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import preprocess_input

#  Rutas del dataset
train_data_dir = "dataset/train"
validation_data_dir = "dataset/valid"

#  Parámetros
width_shape, height_shape = 224, 224
batch_size = 32
num_classes = 20
epochs = 40

#  Data augmentation 
train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# 🖼 Generadores de imágenes
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    class_mode='categorical'
)

#  Número real de imágenes
nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples

#  Entrada del modelo
image_input = Input(shape=(width_shape, height_shape, 3))

#  Cargar modelo base VGG16
base_model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
last_layer = base_model.get_layer('fc2').output

#  Capa de salida personalizada
out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(last_layer)
custom_vgg_model = Model(inputs=image_input, outputs=out)

#  Fine-tuning: descongelar últimas 6 capas
for layer in custom_vgg_model.layers[:-6]:
    layer.trainable = False
for layer in custom_vgg_model.layers[-6:]:
    layer.trainable = True

#  Compilar modelo
custom_vgg_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

#  Callbacks
def lr_schedule(epoch):
    initial_lr = 0.0001
    drop = 0.5
    epochs_drop = 10
    return initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

lr_scheduler = LearningRateScheduler(lr_schedule)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/mejor_modelo.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

#  Entrenamiento
model_history = custom_vgg_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[lr_scheduler, early_stop, checkpoint]
)

print("✅ Entrenamiento finalizado. El mejor modelo fue guardado en: models/mejor_modelo.keras")
