import os                                                                   # Allow interaction with the operating system, such as navigating file directories, creating folders, or manipulating paths.
import glob                                                                 # Used for pattern matching in file names. It can retrieve files or directories that match a specified pattern using wildcards.
import numpy as np                                                          # Library for numerical computing in Python. It's used for handling arrays, performing mathematical operations, and working with matrices.
import tensorflow as tf                                                     # Open-source library used for building and training machine learning models, especially deep learning models. 'tf' is the common alias for TensorFlow.
import splitfolders                                                         # Library that helps split datasets into training, validation, and test sets. It automatically organizes files into respective folders.

def load_dataset(input_folder, dim, output_folder=None, seed=1337, execute_split=False):
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'DataSET')
        
## Check if the user wants to execute the split command
    if execute_split:
        print('Splitting the dataset...')
        splitfolders.ratio(input_folder, output_folder, seed=seed, ratio=(.6, .2, .2), group_prefix=None)
    else:
        print('Skipping dataset splitting...')

## Load the training data
    print('...Scan data...')
    MACp0_train = glob.glob(output_folder + '\\train\\MACp0\\*.tif')
    MACp1_train = glob.glob(output_folder + '\\train\\MACp1\\*.tif')
    MACp2_train = glob.glob(output_folder + '\\train\\MACp2\\*.tif')
    MON_train   = glob.glob(output_folder + '\\train\\MON\\*.tif')

## Load the validation data
    MACp0_val = glob.glob(output_folder + '\\val\\MACp0\\*.tif')
    MACp1_val = glob.glob(output_folder + '\\val\\MACp1\\*.tif')
    MACp2_val = glob.glob(output_folder + '\\val\\MACp2\\*.tif')
    MON_val   = glob.glob(output_folder + '\\val\\MON\\*.tif')

    dataTRAIN = []
    labelsTRAIN = []
    dataVAL = []
    labelsVAL = []

    print(f'...Resizing images to {dim}x{dim}...')

## Process the training data
    for i in MACp0_train:
        image = tf.keras.preprocessing.image.load_img(i,                        # Load the image from the file path 'i' using TensorFlow's 'load_img' function.
                                                      color_mode='grayscale',   # 'color_mode='grayscale'' ensures the image is loaded as a grayscale image.
                                                      target_size=(dim, dim)    # 'target_size=(dim, dim)' resizes the image to the specified dimensions (dim x dim).
                                                      )
        image = np.array(image)                                             # Convert the loaded image into a NumPy array, which is a format suitable for further processing and feeding into a neural network.
        dataTRAIN.append(image)                                             # Append the processed image (now in array form) to the 'dataTRAIN' list, which is used to store the training data.
        labelsTRAIN.append(0)                                               # Append the label '0' to the 'labelsTRAIN' list to indicate that this image belongs to the "Macrophages phenotype 0" class.

    for i in MACp1_train:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataTRAIN.append(image)
        labelsTRAIN.append(1)                                               # Append the label '1' to the 'labelsTRAIN' list to indicate that this image belongs to the "Macrophages phenotype 1" class.

    for i in MACp2_train:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataTRAIN.append(image)
        labelsTRAIN.append(2)                                               # Append the label '2' to the 'labelsTRAIN' list to indicate that this image belongs to the "Macrophages phenotype 2" class.

    for i in MON_train:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataTRAIN.append(image)
        labelsTRAIN.append(3)                                               # Append the label '3' to the 'labelsTRAIN' list to indicate that this image belongs to the "Monocyte" class.


## Process the validation data
    for i in MACp0_val:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataVAL.append(image)
        labelsVAL.append(0)                                                 # label 0 for  Macrophages phenotype 0

    for i in MACp1_val:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataVAL.append(image)
        labelsVAL.append(1)                                                 # label 1 for Macrophages phenotype 1

    for i in MACp2_val:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataVAL.append(image)
        labelsVAL.append(2)                                                 # label 2 for Macrophages phenotype 2

    for i in MON_val:
        image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))
        image = np.array(image)
        dataVAL.append(image)
        labelsVAL.append(3)                                                 # label 3 for Monocytes


## Convert lists to numpy arrays
    dataTRAIN = np.array(dataTRAIN)     	                                # Convert the 'dataTRAIN' list (containing training image data) to a NumPy array for efficient processing.
    labelsTRAIN = np.array(labelsTRAIN)                                     # Convert the 'labelsTRAIN' list (containing the training labels) to a NumPy array.
    dataVAL = np.array(dataVAL)                                             # Convert the 'dataVAL' list (containing validation image data) to a NumPy array.
    labelsVAL = np.array(labelsVAL)                                         # Convert the 'labelsVAL' list (containing the validation labels) to a NumPy array.

## Reshape and normalize the data
    # Reshape the training data ('dataTRAIN') into the shape (num_samples, dim, dim, 1), where the last dimension represents a single grayscale channel.
    # Normalize the pixel values to a range of 0 to 1 by dividing by 255.0, ensuring the data is ready for input into the neural network.
    x_train = dataTRAIN.reshape((dataTRAIN.shape[0], dim, dim, 1)).astype('float32') / 255.0    
    # Similarly, reshape and normalize the validation data ('dataVAL') in the same way as the training data.
    x_val = dataVAL.reshape((dataVAL.shape[0], dim, dim, 1)).astype('float32') / 255.0

    y_train = labelsTRAIN                                                   # Assign the training labels (already in NumPy array format) to 'y_train'.
    y_val = labelsVAL                                                       # Assign the validation labels (already in NumPy array format) to 'y_val'.

    return (x_train, y_train), (x_val, y_val)                               # Return the processed training and validation data as tuples (features and labels) for further use in training the neural network.