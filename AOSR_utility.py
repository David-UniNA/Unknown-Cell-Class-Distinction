import numpy as np                                                          # Imports the NumPy library, which is essential for numerical operations and handling arrays efficiently in Python.
import tensorflow as tf                                                     # Imports TensorFlow, a powerful open-source framework for machine learning and deep learning, allowing the creation and training of neural networks.
from sklearn.ensemble import IsolationForest                                # Imports the Isolation Forest algorithm from the scikit-learn library, which is used for outlier detection in datasets.


## Enrichment with Isolation Forest (IF) -----------------
def sample_enrichment_IF(r_seed, target_data, sample_size):
    np.random.seed(r_seed)                                                  # start with random value
    domain_max = target_data.max(axis=0)                                    # get MAX value
    domain_min = target_data.min(axis=0)                                    # get MIN value
    domain_dim = target_data.shape[1]                                       # get sample types (number of different classes)

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
    def __init__(AOSR, model, x_q, x_w, z_p_X, beta, k):
        super().__init__(name='pq_risk')
        AOSR.model = model
        AOSR.x_q = x_q                                                      # T
        AOSR.x_w = x_w                                                      # W
        AOSR.k = k                                                          # LABELunknown (default = 4)
        AOSR.z_p_X = z_p_X                                                  # Encoder_INI
        AOSR.beta = beta                                                    # Defines the outlier ratio
 
    def call(AOSR, y_true, y_pred):
        # Compute Rs_all_hat
        Rs_all_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Compute Rt_k_hat
        y_t_pred = AOSR.model(AOSR.x_q)
        y_true_q = tf.zeros_like(AOSR.x_w, dtype=tf.int32) + AOSR.k
        Rt_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_q, y_t_pred)
        
        # Apply the weight and reduce mean
        Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(AOSR.x_w, dtype=tf.float32), Rt_k_hat)
        Rt_k_hat = tf.reduce_mean(Rt_k_hat)
        
        # Compute num_out and safeguard against zero
        num_out = tf.math.argmax(AOSR.model(AOSR.z_p_X), axis=1)
        num_out = tf.reduce_sum(tf.cast(tf.equal(num_out, AOSR.k), tf.float32))
        num_out = tf.maximum(num_out, tf.constant(1.0))                     # Prevent division by zero
        
        # Calculate outlier
        outlier = tf.cast(AOSR.z_p_X.shape[0], tf.float32) * AOSR.beta
        
        # Return the combined loss
        return Rs_all_hat + (outlier / num_out) * Rt_k_hat