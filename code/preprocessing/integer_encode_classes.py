from sklearn.preprocessing import LabelEncoder

def integer_encode_classes(y_train_raw):
    # Integer-encode the string labels so that class one
    # is 0, class two is 1, and class three is 2, etc.
    encoder = LabelEncoder()
    y_train_integers = encoder.fit_transform(y_train_raw)
    print('Original class labels:', encoder.classes_)
    return y_train_integers
