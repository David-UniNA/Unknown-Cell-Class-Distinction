"""
Copyright (c) [2024] [Dannhauser David, University of Naples, Federico II]
This script is licensed under the MIT License. 
You may obtain a copy of the License at: https://opensource.org/licenses/MIT

Description:
    This script implements a Convolutional Neural Network (CNN) for open-set cell image classification.
    It utilizes tensorFlow and keras for model training and evaluation. The script loads a model which 
    was produced by "AOSR_ClosedSet_Training_Dannhauser".

Working flow:
    - load a dataset for known and unknown (THP) image classes
    - load the closed-set model
    - use the Ecoder information of the closed-set model as input for the open-set recognition
      Initialize the auxillary domain
    - define range of beta (hyperparameter for open-set recognition)
    - define a open-set Neural Network
    - train open-set model with low epoch number
    - train open-set model again with AOSR loss and high epoch number
    - classify image classes

Dependencies:
    - tensorFlow
    - keras
    - os
    - glob
    - numpy
    - pandas
    - matplotlib.pyplot
    - scikit-learn
    - seaborn

Calls
    Load_Dataset.py
    Metric_plots.py
    AOSR_plots.py
    AOSR_utility.py

Citations:
    If you use any methods, datasets, or specific algorithms, please cite:
    [Cioffi G, Dannhauser D, et al. Biomed Opt Express., 14(10):5060-5074, 2023. doi: 10.1364/BOE.492028]

Contact:
    For any questions or inquiries, please contact [david.dannhauser@unina.it].

Version: 1.0.0
"""

import glob                                                                 # Imports the glob module for Unix-style pathname pattern expansion, allowing for the retrieval of files matching specified patterns (e.g., file extensions).
from sklearn.metrics import accuracy_score                                  # Imports the accuracy_score function from scikit-learn, which calculates the accuracy of a classification model by comparing predicted and true labels.
import numpy as np                                                          # Imports the NumPy library for efficient numerical operations, especially for working with arrays and matrices.
import tensorflow as tf                                                     # Imports TensorFlow, a powerful library for building and training machine learning and deep learning models.
import matplotlib.pyplot as plt                                             # Imports the matplotlib library for creating static, animated, and interactive visualizations in Python.
import pandas as pd                                                         # Imports the pandas library for data manipulation and analysis, providing data structures like DataFrames for handling structured data.
from Load_Dataset import load_dataset                                       # Imports the load_dataset function from the Load_Dataset module, which is responsible for loading and processing image datasets.
from tSNE_plot import plot_tsne                                             # Imports the plot_tsne function from the tSNE_plot module, used for visualizing high-dimensional data using t-SNE (t-distributed Stochastic Neighbor Embedding).
from AOSR_plots import plot_AOSR_results                                    # Imports the plot_AOSR_results function from the AOSR_plots module, which generates visualizations related to the AOSR results.
from AOSR_utility import sample_enrichment_IF, aosr_risk                    # Imports the sample_enrichment_IF and aosr_risk functions from the AOSR_utility module, which provide utilities for sampling and evaluating risks in the AOSR context.


## Initialization...
test_DIR = ('C:\\Python\\Open_set_AOSR\\Unknown_Cell_Class_Distinction')    # Set the directory path for the test samples.
input_folder  = ('C:\\Python\\Open_set_AOSR\\Unknown_Cell_Class_Distinction\\AOSR_dataset') # Define the input folder containing the dataset.
basic_model = tf.keras.models.load_model(test_DIR + '\\CNN__AOSR_100_class4_3.keras')   # Load the pre-trained closed-set model from the specified directory.
dim = 100                                                                   # Set the image dimension for resizing images. (should be the same of the pre-trained closed-set training)
beta_values = np.linspace(0.0, 1, 3)                                       # Generate an array of beta values from 0.0 to 0.5
beta_show = 0.05                                                            # Beta value used for oresented AOSR prediction
E1 = 20                                                                     # Epoch number of 1st AOSR prediction round
E2 = 100                                                                    # Epoch number of 2nd AOSR prediction round
LABELunknown = 4                                                            # Label number of Unknowns
IF = 1000                                                                   # Sample size of Isolation Forest
(X_INI, Y_INI),(X_val, Y_val) = load_dataset(input_folder, dim, execute_split=False)# Load the dataset. Set execute_split to True if you want to split the dataset, otherwise False.
unknownDATA = []                                                            # Initialize a list to hold the loaded images.
unknownLABEL = []                                                           # Initialize a list to hold the corresponding labels.
accuracies = []                                                             # Initialize a list to store accuracy results


## Import unknown samples
THP = glob.glob(input_folder + '\\output\\train\\THP\\*.tif')               # Retrieve all .tif files from the specified path for unknown samples (THP).

for i in THP:                                                               # Loop through the list of unknown sample file paths.
    image = tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', target_size=(dim, dim))  # Load each image, converting it to grayscale and resizing it to the specified dimension.
    image = np.array(image)                                                 # Convert the image to a NumPy array.
    unknownDATA.append(image)                                               # Append the image to the unknown data list.
    unknownLABEL.append(LABELunknown)                                       # Append the label for unknown cells (default = 4).

## Convert lists to NumPy arrays for easier manipulation.
X_THP = np.array(unknownDATA)                                               # Convert the list of unknown images to an array.
Y_THP = np.array(unknownLABEL)                                              # Convert the list of labels to an array.

## Reshape and normalize the data: reshape the data to include a channel dimension 
X_THP = X_THP.reshape((X_THP.shape[0], dim, dim, 1)).astype('float32') / 255.0  # (1 for grayscale) and normalize the pixel values to be between 0 and 1.

## Feature Encoding
tf.random.set_seed(0)                                                       # Set a random seed for reproducibility in TensorFlow operations.

## Create a new Keras model that takes the same input as the basic_model but outputs the second-to-last layer's activations (feature vector).
Encoder = tf.keras.models.Model(inputs=basic_model.layers[0].input,
                                outputs=basic_model.layers[-2].output)  

## Use the Encoder model to generate encoded feature vectors for the training dataset (X_INI) and the unknown dataset (X_THP).
Encoder_INI = Encoder.predict(X_INI)                                        # Encoded features for the training dataset.
Encoder_THP = Encoder.predict(X_THP)                                        # Encoded features for the unknown dataset.
Encoder_ALL = np.concatenate((Encoder_INI, Encoder_THP))                    # Combine Encoder data of initial and Unknowns

## Check for NaN (Not a Number) values in the encoded features or labels.
if np.any(np.isnan(Encoder_INI)) or np.any(np.isnan(Y_INI)):
    print("Warning: NaN values found in input data!")                       # Print warning if NaNs are detected.

## Check for infinite values in the encoded features or labels.
if np.any(np.isinf(Encoder_INI)) or np.any(np.isinf(Y_INI)):
    print("Warning: Infinite values found in input data!")                  # Print warning if infinities are detected.

## Assertions to ensure there are no NaN or infinite values in the encoded features and labels.
assert not np.any(np.isnan(Encoder_INI)), "Encoder_INI contains NaNs"       # Assert Encoder_INI does not contain NaNs.
assert not np.any(np.isinf(Encoder_INI)), "Encoder_INI contains infinite values" # Assert Encoder_INI does not contain infinite values.
assert not np.any(np.isnan(Y_INI)), "Y_INI contains NaNs"                   # Assert Y_INI does not contain NaNs.
assert not np.any(np.isinf(Y_INI)), "Y_INI contains infinite values"        # Assert Y_INI does not contain infinite values.


# ----------------------------------------------------------
# --- t-SNE plot initial & initial+Unknown data ------------
tsne = plot_tsne(X_INI, Y_INI, title="T-SNE projection of initial-Dataset", 
                 n_components=3, perplexity=100.0, early_exaggeration=10.0, random_state=123, 
                 save_path='./tSNE_INI_0__P100__EE10__LR100_N1000.png')
#----------------------------------------------------------
# Concatenate the original dataset X and unknown dataset X_THP for combined analysis
x_ALL = np.concatenate((X_INI, X_THP))                                      # Combine training and unknown samples
y_ALL = np.concatenate((Y_INI, Y_THP))                                      # Combine their corresponding labels
tsne = plot_tsne(x_ALL, y_ALL, title="T-SNE projection of initial-Dataset with Unknowns", 
                 n_components=3, perplexity=100.0, early_exaggeration=10.0, random_state=123, 
                 save_path='./tSNE_INI_UNK_0__P100__EE10.png')
#----------------------------------------------------------
#----------------------------------------------------------


## Initialize auxiliary domain
T, W = sample_enrichment_IF(0, Encoder_INI, int(Encoder_INI.shape[0]))      # Sample enrichment with weight estimation algorithm
W = 1 - W                                                                   # Invert values to adjust the weight for the unknown samples
tau = np.sort(W)[int(W.shape[0] * .1)]                                      # Calculate the tau value as the 10th percentile of W
cond1 = (W > tau)                                                           # Identify unknown samples which have a weight greater than tau
cond2 = (W <= tau)                                                          # Identify known samples which have a weight less than or equal to tau
W[cond2] = 0                                                                # Set weights of known samples to 0 to ignore them
W = W * (W.shape[0] / W.sum())                                              # Normalize weights to ensure they sum to the number of samples


## Open-set learning
# Define a simple neural network model for AOSR
AOSR = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5),                                           # Output layer with 5 units (for 5 classes)
        tf.keras.layers.BatchNormalization(),                               # Normalization layer for better training stability
        tf.keras.layers.Activation(activation='softmax')                    # Softmax activation for probability output
    ])


# ----------------------------------------------------------
# --- t-SNE plot of Encoder --------------------------------
# Plot t-SNE for the initial dataset using the defined plot_tsne function
tsne = plot_tsne(Encoder_ALL, y_ALL, title="T-SNE projection of Encoder-data", 
                 n_components=3, perplexity=100.0, early_exaggeration=10.0, random_state=123, 
                 save_path='./tSNE_ENCODER__P100__EE10__.png')
#----------------------------------------------------------
#----------------------------------------------------------


# Compile the model with Adam optimizer and sparse categorical crossentropy loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)      # Initial learning rate
AOSR.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
# Fit the model to the encoder output for initial training
AOSR.fit(Encoder_INI, Y_INI, epochs=E1)                                     # 1st phase of AOSR training

sample_test, _ = sample_enrichment_IF(0, Encoder_INI, IF)                   # Sample a subset (1000) of the encoded feature vectors (Encoder_X) for testing.
y_pred = AOSR.predict(sample_test)                                          # Use the trained AOSR model to predict the labels for the selected test samples (sample_test).
y_pred_lab = y_pred.argmax(axis=1)                                          # Convert the predicted probability distributions to class labels. ('argmax(axis=1)' selects the class with the highest probability for each sample.

## Plot dataset
plot_AOSR_results(Test=sample_test, Pred=y_pred_lab, Encoder=None,        Encoder_Pred=None,  T=None, W=None, plot_type='closed_set')
plot_AOSR_results(Test=sample_test, Pred=y_pred_lab, Encoder=Encoder_INI, Encoder_Pred=Y_INI, T=None, W=None, plot_type='uniform_classification')
plot_AOSR_results(Test=None,        Pred=None,       Encoder=None,        Encoder_Pred=None,  T=T,    W=W,    plot_type='openset_enrichment')
plot_AOSR_results(Test=None,        Pred=None,       Encoder=Encoder_INI, Encoder_Pred=Y_INI, T=None, W=None, plot_type='open_set')

## Loop over beta values for experimentation
for beta in beta_values:
    print(f'Running for beta = {beta}')
    tf.random.set_seed(0)                                                   # Set random seed for reproducibility
    # Update optimizer and compile the model with custom risk loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)  # Reduced learning rate for fine-tuning
    AOSR.compile(optimizer=optimizer, loss=aosr_risk(AOSR, T, W, Encoder_INI, beta, LABELunknown), metrics=['accuracy'])
    
    fit_params = {'epochs': E2, 'batch_size': 32}                           # Set training parameters
    history = AOSR.fit(Encoder_INI, Y_INI, **fit_params)                    # Train the model with updated settings

    # Predict and calculate accuracy on the training data
    SALS_pred = AOSR.predict(Encoder_INI)                                   # Get predictions
    SALS_pred = SALS_pred.argmax(axis=1)                                    # Convert probabilities to class labels
    accuracy = accuracy_score(Y_INI, SALS_pred)                             # Calculate accuracy score
    accuracies.append(accuracy)                                             # Store accuracy for this beta value

    print(f'Accuracy for beta={beta}: {accuracy * 100:.2f}%')               # Print accuracy result


## Save results to a CSV file
results_df = pd.DataFrame({'Beta': beta_values, 'Accuracy': accuracies})    # Create a DataFrame with beta values and corresponding accuracies.
results_df.to_csv('Beta_vs_Accuracy.csv', index=False)                      # Save the DataFrame to a CSV file without the index.

## Plot beta vs accuracy
plt.plot(beta_values, accuracies, marker='o')                               # Plot beta values on the x-axis and accuracies on the y-axis, using circles as markers.
plt.title('Beta vs Accuracy')                                               # Set the title of the plot.
plt.xlabel('Beta')                                                          # Label for the x-axis.
plt.ylabel('Accuracy')                                                      # Label for the y-axis.
plt.grid(True)                                                              # Enable grid lines for better readability of the plot.
plt.show()                                                                  # Display the plot.
print(f'Accuracy for beta={beta}: {accuracy * 100:.2f}%')                   # Print accuracy result


# Update optimizer and compile the model with custom risk loss function using a selected beta
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)      # Reduced learning rate for fine-tuning
AOSR.compile(optimizer=optimizer, loss=aosr_risk(AOSR, T, W, Encoder_INI, beta_show, LABELunknown), metrics=['accuracy'])    # Beta manually selected
fit_params = {'epochs': E2, 'batch_size': 32}                               # Set training parameters
history = AOSR.fit(Encoder_INI, Y_INI, **fit_params)                        # Train the model with updated settings

# Prediction accuracy for selected beta
SALS_pred = AOSR.predict(Encoder_INI)                                       # Prediction of SALS data
SALS_pred = SALS_pred.argmax(axis=1)                                        # Round the outcome.
accuracy = accuracy_score(Y_INI, SALS_pred)                                 # Calculate accuracy score
accuracies.append(accuracy)                                                 # Store accuracy for this beta value

# 
sample_test, _ = sample_enrichment_IF(0, Encoder_ALL, IF)
y_pred = AOSR.predict(sample_test)                                          # Use the trained AOSR model to predict the labels for the selected test samples.
y_pred_lab = y_pred.argmax(axis=1)                                          # Convert the predicted probability distributions to class labels.
plot_AOSR_results(Test=sample_test, Pred=y_pred_lab, Encoder=None, Encoder_Pred=None, T=None, W=None, plot_type='classification')
print('AOSR prediction of Unknown samples...')
THP_pred  = AOSR.predict(Encoder_THP)                                       # Use the trained AOSR model to predict the labels for Unknown (THP) samples.
THP_pred  = THP_pred.argmax(axis=1)                                         # Convert the predicted probability distributions to class labels.


# ----------------------------------------------------------
# --- t-SNE plot of predicted data -------------------------
tsne = plot_tsne(sample_test, y_pred_lab, title="T-SNE projection of predictions with Unknowns", 
                  n_components=3, perplexity=100.0, early_exaggeration=10.0, random_state=123, 
                  save_path='./tSNE_PRED__P100__EE10.png')
# ----------------------------------------------------------
# ----------------------------------------------------------


# SUMMARIZE OUTCOME ----------------------------
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('---------------------------------------------------------------')
print('AOSR - OUTCOME... ---------------------------------------------')
print('Prediction accuracy of', int(accuracy_score(Y_INI, SALS_pred)*100),'%','for known samples.')
print('---------------------------------------------------------------')
print('Detection of', int((SALS_pred == LABELunknown).sum() / SALS_pred.shape[0]*100),'%','of unknowns in only known samples.')
print('---------------------------------------------------------------')
print('Detection accuracy of', int((THP_pred == LABELunknown).sum() / THP_pred.shape[0]*100),'%',' of unknowns (THP) in only unknown samples.')
print('---------------------------------------------------------------')
print('Process completed and results saved.')                               # Print a completion message.
