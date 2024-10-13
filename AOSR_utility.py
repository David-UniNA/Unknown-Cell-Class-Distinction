import numpy as np                                                          # Imports the NumPy library, which is essential for numerical operations and handling arrays efficiently in Python.
import tensorflow as tf                                                     # Imports TensorFlow, a powerful open-source framework for machine learning and deep learning, allowing the creation and training of neural networks.
from sklearn.ensemble import IsolationForest                                # Imports the Isolation Forest algorithm from the scikit-learn library, which is used for outlier detection in datasets. It is based on the idea of isolating anomalies instead of profiling normal data points.


## Enrichment with Isolation Forest (IF) -----------------
def sample_enrichment_IF(r_seed, target_data, sample_size):
    np.random.seed(r_seed)                                                  # start with random value if 0
    domain_max = target_data.max(axis=0)                                    # get MAX value
    domain_min = target_data.min(axis=0)                                    # get MIN value
    domain_dim = target_data.shape[1]                                       # get sample number

    sample_enri = np.random.random(size=(sample_size, domain_dim))          # create new array
    
    domain_gap  = (domain_max - domain_min) * 1.2                           # gap calculation 
    domain_mean = (domain_max + domain_min) / 2                             # mean value calculation
    
    for dim_idx in range(domain_dim):
        sample_enri[:,dim_idx] = sample_enri[:,dim_idx] * domain_gap[dim_idx] + domain_mean[dim_idx] - domain_gap[dim_idx] / 2
    
    # Outlier sample score given by isolationforest as the sample weight close to 1 - known; close to 0 - unknown class
    clf = IsolationForest(random_state=r_seed, max_samples=0.9).fit(target_data) #max_sample=0.9.... 0,98
    sample_coef = clf.score_samples(sample_enri)                            # sample coefficient from scores
    sample_coef -= sample_coef.min()                                        # Subtract min value
    sample_coef /= sample_coef.max()                                        # Normalize the coef
    print(np.unique(sample_coef).shape)
    return sample_enri, np.squeeze(sample_coef)                             # The output T & W are returned 


## Enrichment with Isolation Forest (IF) -----------------
class aosr_risk(tf.keras.losses.Loss):
    def __init__(AOSR, model, T, W, Encoder_INI, beta, LABELunknown):       # Data preparation, Encoder data = feature space
        super().__init__(name='pq_risk')                                    # Used to initialize the parent class with a specific argument (name='pq_risk'). Super() is used to call a method from a parent class. In the context of class inheritance, it allows you to access methods and properties of the superclass (the parent class) without explicitly naming it.
        AOSR.model = model                                                  # The AOSR model
        AOSR.T = T                                                          # T the enriched sample after Isolation Forest
        AOSR.W = W                                                          # W the weights after Isolation Forest
        AOSR.Enc = Encoder_INI                                              # Encoded features from the training dataset.
        AOSR.beta = beta                                                    # Defines the outlier ratio
        AOSR.Lab = LABELunknown                                             # Label number of unknowns (default = 4)
 
    def call(AOSR, y_true, y_pred):
        # Compute initial Loss
        Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)      # Compute the sparse categorical crossentropy loss between the true labels (y_true) and the predicted labels (y_pred).
        
        # Compute enriched sample Loss with Unknowns
        T_pred = AOSR.model(AOSR.T)                                         # Predict enriched sample
        T_true = tf.zeros_like(AOSR.W, dtype=tf.int32) + AOSR.Lab           # Creat labels with Unknowns
        T_Loss = tf.keras.losses.sparse_categorical_crossentropy(T_true, T_pred)    # Compute the sparse categorical crossentropy loss
        
        # Apply the weight and reduce mean
        T_Loss = tf.math.multiply(tf.convert_to_tensor(AOSR.W, dtype=tf.float32), T_Loss)   # Convert AOSR.W into a float tensor and then multiplies it element-wise by T_Loss
        T_Loss = tf.reduce_mean(T_Loss)                                     # Computes the mean average value of all elements in the tensor T_Loss
        
        # Compute num_out and safeguard against zero
        Num_outlier = tf.math.argmax(AOSR.model(AOSR.Enc), axis=1)          # Computes the indices of the maximum predicted probabilities for each input sample, determining the predicted class labels based on the model’s outputs.
        Num_outlier = tf.reduce_sum(tf.cast(tf.equal(Num_outlier, AOSR.Lab), tf.float32))   # Counts the number of correct predictions made by a model by comparing predicted labels to true labels.
        Num_outlier = tf.maximum(Num_outlier, tf.constant(1.0))             # Prevent division by zero. It creates a new tensor where each element is the maximum of the corresponding element in num_out and 1.0
        
        # Calculate outlier
        outlier = tf.cast(AOSR.Enc.shape[0], tf.float32) * AOSR.beta        # Outlier calculation depending on beta
        
        # Return the combined loss
        return Loss + (outlier / Num_outlier) * T_Loss                      # Output of aosr_risk