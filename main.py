from django.shortcuts import redirect, render


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, save_model, load_model

import urllib.request
from PIL import Image
import pathlib
import os
# Create your views here.

#View is being requested
def index(request):
    return render(request,'detector_app/index.html')

def results(request):
    context = {}
    if request.method == 'POST':
        image_url = request.POST.get("url-input")
        feedModel(image_url)
        context["image_url"] = image_url
    return render(request,'detector_app/results.html', context)

def feedModel(image_url):
    #Setup data path directory
    data_dir = pathlib.Path('../ISIC_2019_Training_Input')

    #Parameters for loading dataset from images
    batch_size = 32
    img_height = 224
    img_width = 224 

    #Configure training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.125, #Use 20% of data for testing
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

    #Configure testing data
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.125,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

    #Get names of subfolders
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(class_names)

    #Configure cache and prefetch
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #Rescaling layer to convert RGB values [0, 255] to [0, 1]
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3))

    #Creating the CNN - Convolutional Neural Network
    model = Sequential([
    # normalization_layer,
    # layers.Conv2D(16, 3, padding='same', activation='relu'),  #Convolution layer
    # layers.MaxPooling2D(),    #Pooling layer
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Flatten(),
    # layers.Dropout(0.3),
    # layers.Dense(128),
    # layers.Dense(num_classes, activation='softmax'),
    ])
    mobile = keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    x = layers.Dropout(0.25)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=mobile.input, outputs=predictions)
  
    METRICS = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    #Calculate class weights
    class_weights = {0: 12875/867, 1: 12875/3323, 2: 12875/1800, 3: 12875/2624, 
    4: 12875/239, 5: 12875/12875, 6: 12875/4522, 
    7: 12875/628, 8: 12875/263}

    #Compiling the model
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    #Configure checkpoint callback
    checkpoint_path = "trained_weights/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.load_weights(checkpoint_path)

    #Training the model
    epochs=100  #Rounds of training/testing 
    # history = model.fit(
    # train_ds,
    # validation_data=test_ds,
    # epochs=epochs,
    # callbacks=[cp_callback],
    # class_weight = class_weights
    # )

    predict_url = image_url
    predict_path = tf.keras.utils.get_file(image_url[-10:], origin=predict_url)

    img = keras.preprocessing.image.load_img(
        predict_path, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    #img_array = img_array/255.


    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


  


