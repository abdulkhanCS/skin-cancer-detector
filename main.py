import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, save_model, load_model


import pathlib

import urllib.request
from PIL import Image

#Setup data path directory
data_dir = pathlib.Path('./ISIC_2019_Training_Input')

#Display an image
# img = PIL.Image.open(str(vehicles[0]))
# img.show()

#Parameters for loading dataset from images
batch_size = 32
img_height = 240
img_width = 240

#Configure training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, #Use 20% of data for testing
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Configure testing data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Get names of subfolders
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

#Show first 9 images from the training data set
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#     plt.show()

#Configure cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Rescaling layer to convert RGB values [0, 255] to [0, 1]
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3))

#Creating the CNN - Convolutional Neural Network
model = Sequential([
  normalization_layer,
  layers.Conv2D(16, 3, padding='same', activation='relu'),  #Convolution layer
  layers.MaxPooling2D(),    #Pooling layer
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#Compiling the model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#View model architecture
#model.summary()

#Configure checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model.load_weights(checkpoint_path)

#Training the model
epochs=11   #Rounds of training/testing 
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs,
  callbacks=[cp_callback]
  )

#Testing with new single input image
#model.load_weights(checkpoint_path)

while True:

  predict_url = input("Enter a url: ")

  if predict_url == 'q':
    break
  predict_path = tf.keras.utils.get_file("Image", origin=predict_url)

  urllib.request.urlretrieve(predict_url, "image")
  img = Image.open("image")
  img.show()

  img = keras.preprocessing.image.load_img(
      predict_path, target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  


