import matplotlib.pyplot as plt                                             # Imports the pyplot module from Matplotlib, a popular library for creating static, animated, and interactive visualizations in Python. 
                                                                            # This module is commonly used for plotting graphs and visualizing data, such as training and validation metrics during model evaluation.

def plot_metrics(history, metric, filename):
    
    """
    Plots the specified metric (e.g., loss or accuracy) over epochs for both training 
    and validation datasets, and saves the plot to a specified file.

    Parameters:
        history (tf.keras.callbacks.History): The history object returned by 
            the model training process, which contains the metric values over epochs.
        metric (str): The name of the metric to plot. This should be either 'loss' 
            or 'accuracy' (or any other metric defined during model compilation).
        filename (str): The path and filename where the plot image will be saved.

    Example:
        plot_metrics(history, 'loss', 'loss_plot.jpg')
    """

    plt.plot(history.history[metric], label='train ' + metric)                  # Plot training metric against epochs
    plt.plot(history.history['val_' + metric], label='validation ' + metric)    # Plot validation metric against epochs
    plt.legend()                                                                # Add a legend to distinguish between training and validation metrics
    plt.savefig(filename)                                                       # Save the plot to the specified filename
    plt.show()                                                                  # Display the plot
    plt.close()                                                                 # Close the current figure to free up memory
