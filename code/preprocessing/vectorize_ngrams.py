from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_ngrams(X_train_sequences,
                     X_test_sequences,
                     y_train_integers,
                     max_features,
                     token_mode,
                     ngram_range,
                     min_df):
    """Construct n-grams from text sequences and vectorize them.
    
    Also score the importance of the vectors via term frequency and
    select the most important features, if a maximum is set.

    Based on Google's `ngram_vectorize` function:
    https://developers.google.com/machine-learning/guides/text-classification/step-3.

    Parameters
    ----------
    X_train_sequences : list
        A list of string sequences (sentences) from the train dataset.
    X_test_sequences : list
        A list of string sequences from the test dataset.
    y_train_integers : numpy.ndarray
        A numpy vector of integer-encoded classes in the shape of
        (num_samples,).
    max_features : int
        The maximum number of n-gram features to include in the bag
        of words; the most important n-grams by term frequency are
        kept first. If set to None, then all n-grams are included.
    token_mode : string
        The mode of tokenization: Either 'word' for word-level or 'char'
        for character-level tokenization.
    ngram_range : tuple
        A tuple of two integers for the range of n-grams to include in
        the bag of words, such as (1, 2) for unigrams and bigrams.
    min_df : int
        The minimum document frequency (number of times a word appears
        across documents in the corpus) required to include a word token
        in the bag of words.

    Returns
    -------
    results : tuple
        - X_train_tokenized : numpy.ndarray
            A 2D integer tensor of shape (num_train_samples, maxlen).
        - X_test_tokenized : numpy.ndarray
            A 2D integer tensor of shape (num_test_samples, maxlen).
    """

    kwargs = {
        # Google Text Classification settings
        'encoding': 'utf-8',
        'decode_error': 'replace',
        'strip_accents': 'unicode',
        'analyzer': token_mode,
        'ngram_range': ngram_range,
        'min_df': min_df,
        'dtype': 'int32',

        # Our own settings
        # We use a custom tokenizer in order to treat punctuation as tokens
        'tokenizer': word_tokenize,
        'lowercase': False,
        'stop_words': None, # Do not remove stopwords
        # Only consider the top `max_features` ordered by term frequency 
        # across the corpus
        'max_features': max_features,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    
    # Learn the vocabulary and idf from the training dataset
    vectorizer.fit(X_train_sequences)
    
    # Encode the documents into a `scipy.sparse.csr.csr_matrix` so that
    # each position in the row vector has the TF-IDF for that n-gram
    X_train_tokenized = vectorizer.transform(X_train_sequences)
    X_test_tokenized = vectorizer.transform(X_test_sequences)
    
    num_features = X_train_tokenized.shape[1]
    print(f'Found {num_features} unique unigrams and bigrams.')

    # If needed: Convert from a `scipy.sparse.csr.csr_matrix` to a
    # `numpy.ndarray` by appending `.toarray()` to each of the following:
    return X_train_tokenized, X_test_tokenized
