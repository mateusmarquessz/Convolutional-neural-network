from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator


train_directory = '/content/drive/MyDrive/Datasets/Carros/treino'
test_directory = '/content/drive/MyDrive/Datasets/Carros/test'
val_directory = '/content/drive/MyDrive/Datasets/Carros/val'


categories = os.listdir(train_directory)
print(str(len(categories)),'CATEGORIES are ', categories)

category_count = len(categories)

def preprocess_image(image):
    if image.shape[-1] == 1:  # Verifique se a imagem está em escala de cinza
        # Converta a imagem de escala de cinza para escala RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        rgb_image = image
    return rgb_image

augmented_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_image
)

general_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_image
)

train_generator = general_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb'  # Defina o modo de cor como RGB
)

valid_generator = general_datagen.flow_from_directory(
    val_directory,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb'  # Defina o modo de cor como RGB
)

test_generator = general_datagen.flow_from_directory(
    test_directory,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb'  # Defina o modo de cor como RGB
)

train_groups = len(train_generator)
valid_groups = len(valid_generator) # validation_step

print(f"Train groups: {train_groups}")
print(f"Validation groups: {valid_groups}")

def conv_layer(inputs, filters, kernel_size=3, padding="valid"):
    x = layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, use_bias = False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x
# pooling layer i added dropout cause it help my model to reduce the overfitting
def pooling_layer(inputs, pool_size = 2, dropout_rate=0.5):
    x = layers.MaxPooling2D(pool_size = pool_size)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    return x

# this dense layer i will not only use it for my base model i will use it in the pretrained model too
def dense_layer(inputs, out, dropout_rate = 0.5):
    x = layers.Dense(out)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    return x


keras.backend.clear_session()

inputs = keras.Input(shape = (224, 224, 1))

x = conv_layer(inputs, 64, padding = "same")  # 224x224
x = conv_layer(x, 64)                         # 222x222
x = pooling_layer(x)                          # 111x111

x = conv_layer(x, 64, padding = "same")       # 111x111
x = conv_layer(x, 64)                         # 109x109
x = pooling_layer(x)                          # 54x54

x = conv_layer(x, 64, padding = "same")       # 54x54
x = conv_layer(x, 64)                         # 52X52
x = pooling_layer(x)                          # 26x26

x = conv_layer(x, 64, padding = "same")       # 26x26

x = layers.Flatten()(x)                       # 26*26*64

x = dense_layer(x, 64)

outputs = layers.Dense(category_count, activation = "softmax")(x)

base_model = keras.Model(inputs, outputs)

base_model.summary()

base_model.compile(optimizer =keras.optimizers.Adam(learning_rate=0.001),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit model
history = base_model.fit(
    train_generator,
    steps_per_epoch = train_groups,
    epochs = 20, # adding more epochs will increase the acc like 1% or 2%
    validation_data = valid_generator,
    validation_steps = valid_groups,
    verbose = 1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 10, restore_best_weights = True),
               keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, patience = 2, verbose = 1),
    keras.callbacks.ModelCheckpoint(
            filepath = "/content/drive/MyDrive/Datasets/rim/intial_model.h5",
            save_best_only = True,
            monitor = "val_loss")
    ])

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]


print(accuracy[19])
#epochs = range(1, len(accuracy) + 1)

#plt.plot(epochs, accuracy, "bo", label = "Trianing accuracy")
#plt.plot(epochs, val_accuracy, "b-", label = "Validation accuracy")
#plt.title("Accuracy on training and validation data")
#plt.legend()
#plt.figure()

#plt.plot(epochs, loss, "bo", label = "Trianing loss")
#plt.plot(epochs, val_loss, "b-", label = "Validation loss")
#plt.title("loss on training and validation data")
#plt.title("loss on training and validation data")
#plt.legend()
#plt.show()



# Carregue o modelo
model = keras.models.load_model("/content/drive/MyDrive/Datasets/rim/intial_model.h5")

# Supondo que você já configurou os geradores de dados
# (train_generator, valid_generator, test_generator)

# Avaliar o modelo no conjunto de teste
test_results = model.evaluate(test_generator)

# A função evaluate retorna uma lista com o valor da função de perda (loss)
# e a métrica definida no modelo (neste caso, a acurácia).
loss, accuracy = test_results

print(f'Perda (Loss): {loss}')
print(f'Acurácia: {accuracy}')

predictions = model.predict(test_generator)

# Converter as previsões em classes (rótulos) usando argmax
predicted_classes = np.argmax(predictions, axis=1)

# Obter os rótulos verdadeiros do conjunto de teste
true_classes = test_generator.classes

# Calcular a precisão
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Precisão: {accuracy}')

from sklearn.metrics import f1_score
import numpy as np

# Realizar previsões no conjunto de teste usando o gerador de dados
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Obter os rótulos verdadeiros do conjunto de teste
true_classes = test_generator.classes

# Calcular o F1-Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f'F1-Score: {f1}')

from sklearn.metrics import confusion_matrix
import numpy as np

# Realizar previsões no conjunto de teste usando o gerador de dados
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Obter os rótulos verdadeiros do conjunto de teste
true_classes = test_generator.classes

# Calcular a matriz de confusão
confusion = confusion_matrix(true_classes, predicted_classes)

print('Matriz de Confusão:')
print(confusion)