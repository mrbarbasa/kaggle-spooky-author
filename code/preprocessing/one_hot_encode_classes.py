from keras.utils import to_categorical

def one_hot_encode_classes(y_train_integers):
    """One-hot encode each integer (representing a class) into a vector
    such as [1 0 0].

    Parameters
    ----------
    y_train_integers : numpy.ndarray
        A numpy vector of integer-encoded classes in the shape of
        (num_samples,).
    
    Returns
    -------
    y_train_encoded : numpy.ndarray
        A numpy matrix of one-hot encoded classes in the shape of
        (num_samples, num_classes).
    """

    y_train_encoded = to_categorical(y_train_integers)
    return y_train_encoded
