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

def one_hot_encode_classes(y_train):
    # Integer-encode the string labels so that class one
    # is 0, class two is 1, and class three is 2, etc.
    encoder = LabelEncoder()
    y_train_integers = encoder.fit_transform(y_train)
    # One-hot encode each integer into a vector such as [1 0 0]
    y_train_encoded = to_categorical(y_train_integers)
    print('Original class labels:', encoder.classes_)
    return y_train_encoded

def load_glove_embeddings(embeddings_path):
    # Load the GloVe embeddings into a dictionary
    # This maps words (as strings) to their vector representation (as float vectors)
    embeddings_index = {}
    f = open(embeddings_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found {} word vectors.'.format(len(embeddings_index)))
    return embeddings_index

def construct_embedding_matrix(word_index, embeddings_index, embedding_dim):
    # Note: Only use this if creating a randomly initialized embedding matrix
    # np.stack() --> 
    #   array([[0.32, 0.7 ],
    #          [0.42, 0.1 ]], dtype=float32)
    # all_embeddings = np.stack(embeddings_index.values())
    # embedding_mean, embedding_std = all_embeddings.mean(), all_embeddings.std()
    # print(embedding_mean, embedding_std)
    
    # Compute the embedding matrix using our training words `word_index` and
    # the pre-trained embeddings `embeddings_index`
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Create an embedding matrix with random initialization for words that aren't in GloVe,
    #   using the mean and stdev of the GloVe embeddings
    # embedding_matrix = np.random.normal(loc=embedding_mean,
    #                                     scale=embedding_std,
    #                                     size=(vocab_size, EMBEDDING_DIM))

    # Loop over each of the first `MAX_FEATURES` words of the `word_index` built from
    # the dataset and retrieve its embedding vector from the GloVe `embeddings_index`
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        # Words not found in the `embeddings_index` will have their vectors in `embedding_matrix`
        # remain as all zeros
        # -- or --
        # remain as a random normalization of the mean and stdev of the GloVe embeddings
    return embedding_matrix, vocab_size
