from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def compute_word_index(X_train_sequences,
                       X_test_sequences,
                       max_features,
                       max_sequence_length):
    """Compute the vocabulary word index and vectorize the train and
    test string sequences.

    Parameters
    ----------
    X_train_sequences : list
        A list of string sequences (sentences) from the train dataset.
    X_test_sequences : list
        A list of string sequences from the test dataset.
    max_features : int
        The maximum number of features (vocabulary words) to include in
        every sequence; the larger (less common) word indices get
        omitted first. If this is None, then no words get omitted.
    max_sequence_length : int
        The maximum length of (number of words in) a sequence before it
        gets truncated; also, if the sequence is shorter, it gets padded.

    Returns
    -------
    results : tuple
        - X_train_tokenized : numpy.ndarray
            A 2D integer tensor of shape (num_train_samples, maxlen).
        - X_test_tokenized : numpy.ndarray
            A 2D integer tensor of shape (num_test_samples, maxlen).
        - word_index : dict
            The computed word index, which maps a word to its index.
    """

    # Only include the top `num_words` most common words
    # `filters=''` means no characters will be filtered from the text
    tokenizer = Tokenizer(num_words=max_features,
                          filters='',
                          lower=False,
                          split=' ',
                          char_level=False,
                          oov_token=None)
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
    # (num_samples, maxlen)
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
    # Note that despite the `num_words` set in the Tokenizer, all unique
    # tokens are stored here as if `num_words` were set to `None`.
    # https://github.com/keras-team/keras/issues/7551
    print('Found {} unique tokens.'.format(len(word_index)))
    return X_train_tokenized, X_test_tokenized, word_index
