"""
Copyright (c) [2024] [Dannhauser David, University of Naples, Federico II]
This script is licensed under the MIT License. 
You may obtain a copy of the License at: https://opensource.org/licenses/MIT

Description:
    This script implements a Convolutional Neural Network (CNN) for closed-set cell image classification.
    It utilizes tensorFlow and keras for model training and evaluation. The classifer model is saved for 
    open-set training.

Working flow:
    - load a dataset of known image classes (MACp0, MACp0, MACp0, MON)
    - define a closed-set Neural Network, which classify the dataset
    - train the classifier and save the model

Dependencies:
    - tensorFlow
    - keras
    - os
    - glob
    - numPy
    - matplotlib.pyplot
    - scikit-learn
    - splitfolders

Calls
    Load_Dataset.py
    Metric_plots.py

Citations:
    If you use any methods, datasets, or specific algorithms, please cite:
    [Cioffi G, Dannhauser D, et al. Biomed Opt Express., 14(10):5060-5074, 2023. doi: 10.1364/BOE.492028]

Contact:
    For any questions or inquiries, please contact [david.dannhauser@unina.it].

Version: 1.0.0
"""

import tensorflow as tf                                                     # TensorFlow includes Keras, the high-level neural networks API.
from Load_Dataset import load_dataset                                       # Import the function to load the training dataset
from Metric_plots import plot_metrics                                       # Import the function to plot the CNN metrics

dim = 100                                                                   # Dimension (pixel) of images (default = 100)
Epo = 10                                                                    # Epochs for the training (default = 10)
Bat = 32                                                                    # Batch size for the training (default - 32)
input_folder = 'C:\\Python\\Open_set_AOSR\\Unknown_Cell_Class_Distinction\\AOSR_dataset'    # Folder where image dataset is located

## Load dataset with optional splitting (split datset in training 60%, testing 20% and validation 20%)
(x_train, y_train), (x_val, y_val) = load_dataset(input_folder, dim, execute_split=False)
print(x_train.shape, y_train.shape)                                         # Check shapes of training data to ensure they are loaded correctly
print(x_val.shape, y_val.shape)                                             # Check shapes of validation data to ensure they are loaded correctly

## Define the parameter grid for the closed-set CNN
filters = 8                                                                 # Number of filters for the Conv2D layers
kernel_size = (5, 5)                                                        # Kernel size for the Conv2D layers
dropout_rate = 0.2                                                          # Dropout rate for regularization

## Define the Convolutional Neural Network (CNN) model
basic_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters, kernel_size, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(dim, dim, 1)), # First Conv2D layer, with regularization to avoid overfitting
    tf.keras.layers.BatchNormalization(),                                   # First Batch normalization layer (Normalize the input layer by adjusting and scaling the activations, which helps stabilize and speed up the training)
    tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"),        # Second Conv2D layer
    tf.keras.layers.BatchNormalization(),                                   # Second Batch normalization Layer
    tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"),        # Third Conv2D layer
    tf.keras.layers.BatchNormalization(),                                   # Third Batch normalization Layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                         # First MaxPooling layer
    tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"),        # Fourth Conv2D layer
    tf.keras.layers.Conv2D(filters, kernel_size, activation="relu"),        # Fifth Conv2D layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                         # Second MaxPooling layer
    tf.keras.layers.Flatten(),                                              # Flatten the feature map into a 1D vector
    tf.keras.layers.Dense(200, activation="relu"),                          # First Dense layer with 200 units (The Dense layer is fully connected, meaning each of the 200 units receives input from all neurons in the previous layer)
    tf.keras.layers.Dropout(dropout_rate, seed=2019),                       # Dropout layer for regularization
    tf.keras.layers.Dense(400, activation="relu"),                          # Second Dense layer with 400 units
    tf.keras.layers.Dropout(dropout_rate, seed=2019),                       # Another Dropout layer
    tf.keras.layers.Dense(600, activation="relu"),                          # Third Dense layer with 600 units
    tf.keras.layers.Dense(4),                                               # Output layer with 4 units for 4 classes
    tf.keras.layers.Activation(activation='softmax')                        # Softmax activation for classification
])
basic_model.summary()                                                       # Display the model architecture
    # L2 regularization 'keras.regularixers.l2(0.01)' adds a penalty to the loss function during training, proportional to the square of the magnitude of the weights. 
    # The goal is to prevent overfitting by discouraging the model from relying on large weights, encouraging smaller, more generalizable weights instead.
    # During training, dropout helps regularize the network by preventing units from co-adapting too much. It randomly drops a certain percentage of neurons (as defined by dropout_rate),
    # forcing the network to learn more robust features. This prevents the model from overfitting on the training data, improving generalization to unseen data.

## Compile the CNN model
basic_model.compile(optimizer='adam',                                       # Define optimizer, default = adam
                    loss='sparse_categorical_crossentropy',                 # Define loss, default = sparse_categorical_crossentropy
                    run_eagerly=True,                                       # Enable eager execution for better debugging
                    metrics=['accuracy'])                                   # Define valutation metrics, default = accuracy

basic_model.optimizer.learning_rate.assign(0.001)                           # Set the learning rate for the optimizer

## Train the CNN model on the training data, validating on the validation data
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(                        # Learning Rate Schedular (Dynamically adjust the learning rate during training to improve convergence)
    monitor='val_loss',                                                     # Monitor validation loss
    factor=0.5,                                                             # Factor by which the learning rate will be reduced
    patience=2,                                                             # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6                                                             # Minimum learning rate
)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(                  # Data Augmentation (Artificially increase the size of the training dataset by applying random transformations, which helps improve the model's generalization)
    rotation_range=0,                                                       # Degree range for random rotations
    width_shift_range=0.01,                                                 # Range for horizontal shifts
    height_shift_range=0.01,                                                # Range for vertical shifts
    zoom_range=0.01,                                                        # Range for random zoom
    horizontal_flip=False                                                   # Randomly flip inputs horizontally
)
history = basic_model.fit(
    datagen.flow(x_train, y_train, batch_size=Bat),                         # Define training dataset
    validation_data=(x_val, y_val),                                         # Define validation dataset
    epochs=Epo,                                                             # Epochs for the training
    callbacks=[lr_scheduler]                                                # Add LR scheduler callback
)
basic_model.save('CNN__AOSR_100_class4_1.keras')                            # Save the trained model to a file

## Plot training and validation loss over epochs
plot_metrics(history, 'loss', 'loss_AOSR_100__1.jpg')                       # CNN Loss plot
plot_metrics(history, 'accuracy', 'accuracy_AOSR_100__1.jpg')               # CNN Accuracy plot
