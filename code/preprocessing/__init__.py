import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def compute_word_index(X_train_sequences,
                       X_test_sequences,
                       max_features,
                       max_sequence_length):
    # Only include the top `num_words` most common words
    tokenizer = Tokenizer(num_words=max_features)
    # Build the word index, requiring a list argument
    tokenizer.fit_on_texts(X_train_sequences)

    # Turn strings into a list of lists of integer indices such as 
    # [[688, 75, 1], [...]]
    X_train_tokenized = tokenizer.texts_to_sequences(X_train_sequences)
    X_test_tokenized = tokenizer.texts_to_sequences(X_test_sequences)

    # Pad with 0.0 before each sequence
    # Remove values, before each sequence, from sequences larger than 
    # `maxlen`
    # Turn a list of integers into a 2D integer tensor of shape
    # (samples, maxlen)
    X_train_tokenized = pad_sequences(X_train_tokenized, 
                                   maxlen=max_sequence_length, 
                                   padding='pre', 
                                   truncating='pre', 
                                   value=0.0)
    X_test_tokenized = pad_sequences(X_test_tokenized, 
                                  maxlen=max_sequence_length, 
                                  padding='pre', 
                                  truncating='pre', 
                                  value=0.0)

    # Recover the computed word index, which appears as 
    # {'necessary': 1234, ...}
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))
    return X_train_tokenized, X_test_tokenized, word_index

# The first line of a fastText embeddings file contains:
# - The number of words in the vocabulary
# - The size of each vector
#
# Otherwise, the rest of a fastText file has the same format as GloVe:
# - Each line contains a string word followed by its vector.
# - Each value is space-separated.
# - Words are ordered by descending frequency.
def load_embeddings(embeddings_path):
    # Load the GloVe or fastText embeddings into a dictionary
    # This maps words (as strings) to their vector representation (as float vectors)
    embeddings_index = {}
    f = open(embeddings_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found {} word vectors.'.format(len(embeddings_index)))
    return embeddings_index

def construct_embedding_matrix(word_index, embeddings_index, embedding_dim):    
    # Compute the embedding matrix using our training words `word_index`
    # and the pre-trained embeddings `embeddings_index`
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Keep track of the number of words not found in the pre-trained embeddings
    num_unknown = 0
    # Loop over each of the first `MAX_FEATURES` words of the `word_index`
    # built from the dataset and retrieve its embedding vector from the
    # pre-trained `embeddings_index`
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # A word in our corpus has been found in the pre-trained embeddings
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # Words not found in the `embeddings_index` will have their
            # vectors in `embedding_matrix` remain as all zeros
            num_unknown += 1
    return embedding_matrix, vocab_size, num_unknown

def integer_encode_classes(y_train_raw):
    # Integer-encode the string labels so that class one
    # is 0, class two is 1, and class three is 2, etc.
    encoder = LabelEncoder()
    y_train_integers = encoder.fit_transform(y_train_raw)
    print('Original class labels:', encoder.classes_)
    return y_train_integers

def one_hot_encode_classes(y_train_integers):
    # One-hot encode each integer into a vector such as [1 0 0]
    y_train_encoded = to_categorical(y_train_integers)
    return y_train_encoded
