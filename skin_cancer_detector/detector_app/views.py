from django.shortcuts import redirect, render

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import pathlib

#Index view is requested
def index(request):
    #Render index view
    return render(request,'detector_app/index.html')

#Results page is requested
def results(request):
    #Create a context dict to pass data to results page
    context = {}
    if request.method == 'POST':
        image_url = request.POST.get("url-input")
        #Feed the keras model the url of the image
        feedModel(image_url)
        context["image_url"] = image_url
    #Render results page with context
    return render(request,'detector_app/results.html', context)

#Machine learning model
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
    #Use 12.5% of data for testing
    validation_split=0.125,
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

    #Create the CNN - Convolutional Neural Network
    mobile = keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    x = layers.Dropout(0.25)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=mobile.input, outputs=predictions)

    #Calculate class weights
    class_weights = {0: 12875/867, 1: 12875/3323, 2: 12875/1800, 3: 12875/2624, 
    4: 12875/239, 5: 12875/12875, 6: 12875/4522, 
    7: 12875/628, 8: 12875/263}

    #Compiling the model
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    #Configure checkpoint callback for saving trained model
    checkpoint_path = "trained_weights/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    #Load the trained model
    model.load_weights(checkpoint_path)

    #Train the model
    epochs=100  #Rounds of training/testing 
    # history = model.fit(
    # train_ds,
    # validation_data=test_ds,
    # epochs=epochs,
    # callbacks=[cp_callback],
    # class_weight = class_weights
    # )

    #Feed the target url to the trained model
    predict_url = image_url
    predict_path = tf.keras.utils.get_file(image_url[-10:], origin=predict_url)

    #Preprocess the image before the model predicts on it
    img = keras.preprocessing.image.load_img(
        predict_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    #Create a batch
    img_array = tf.expand_dims(img_array, 0)

    #Output
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    prediction_confidence = 100 * np.max(score)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(predicted_class, prediction_confidence)
    )


  


