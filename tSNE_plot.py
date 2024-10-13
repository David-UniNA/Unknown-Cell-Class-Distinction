import matplotlib.pyplot as plt                                             # Imports the matplotlib library for creating static, animated, and interactive visualizations in Python.
import pandas as pd                                                         # Imports the pandas library for data manipulation and analysis, providing data structures like DataFrames for handling structured data.
from sklearn.manifold import TSNE                                           # Imports the t-SNE algorithm from the scikit-learn library, used for dimensionality reduction and visualization of high-dimensional datasets.
import seaborn as sns                                                       # Imports the seaborn library for statistical data visualization, built on top of matplotlib and provides a high-level interface for drawing attractive graphics.

def plot_tsne(X_data, Y_data, title, n_components=3, perplexity=100.0, early_exaggeration=10.0, random_state=123, save_path=None):

    """
    Function to plot t-SNE of given data.
    
    Parameters:
    X_data: np.array
        Input data to be transformed with t-SNE.
    Y_data: np.array
        Corresponding labels for the data.
    title: str
        Title of the plot.
    n_components: int
        Number of dimensions for t-SNE (default: 3).
    perplexity: float
        t-SNE perplexity parameter (default: 100.0).
    early_exaggeration: float
        t-SNE early exaggeration parameter (default: 10.0).
    random_state: int
        Random seed for reproducibility (default: 123). When you set random_state=123, the random processes in the code will produce the same result each time you run the code, ensuring consistency across different runs.
    save_path: str or None
        Path to save the plot image, if None the plot is not saved (default: None).
    """
    
    X_reshaped = X_data.reshape(X_data.shape[0], -1)                        # Reshapes the input data into a 2D array for t-SNE processing. (X_data.shape[0] gets the number of samples in the dataset, while -1 acts as a placeholder to automatically infer the number of columns)
        # flatten multi-dimensional data (e.g., images) into 2D data suitable for input into machine learning algorithms that expect a 2D array (samples × features).

    print(f"Reshaped X_data: {X_reshaped.shape}")                           # Prints the new shape of the reshaped data.
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                early_exaggeration=early_exaggeration, verbose=1, random_state=random_state)  # Initializes the t-SNE object with specified parameters.
    
    Z = tsne.fit_transform(X_reshaped)                                      # Applies the t-SNE transformation to the reshaped data.
    
    df = pd.DataFrame()                                                     # Creates an empty DataFrame to hold the transformed data and labels.
    df["y"] = Y_data                                                        # Adds the labels to the DataFrame.
    df["tSNE - comp 1"] = Z[:, 0]                                           # Adds the first t-SNE component to the DataFrame.
    df["tSNE - comp 2"] = Z[:, 1]                                           # Adds the second t-SNE component to the DataFrame.
    
    ## Creates a scatter plot of the first two t-SNE components, coloring the points by their labels.
    sns.scatterplot(x="tSNE - comp 1", y="tSNE - comp 2", hue=df.y.tolist(),# hue = df.y.tolist() is used to specify the color of the points in the scatter plot based on the values in the y column of a DataFrame df.
                    palette=sns.color_palette("hls", len(set(Y_data))),     # Generates a color palette for the unique labels.
                    data=df).set(title=title)                               # Sets the title of the plot.
    
    if save_path:                                                           # Checks if a save path is provided.
        plt.savefig(save_path)                                              # Saves the plot to the specified file path.
    
    plt.show()                                                              # Displays the plot.
