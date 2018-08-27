from keras.utils import to_categorical

def one_hot_encode_classes(y_train_integers):
    # One-hot encode each integer into a vector such as [1 0 0]
    y_train_encoded = to_categorical(y_train_integers)
    return y_train_encoded
