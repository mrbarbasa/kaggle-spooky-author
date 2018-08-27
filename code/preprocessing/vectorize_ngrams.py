from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

# Based on Google's `ngram_vectorize` function:
# https://developers.google.com/machine-learning/guides/text-classification/step-3
def vectorize_ngrams(X_train_sequences,
                     X_test_sequences,
                     y_train_integers,
                     max_features,
                     token_mode,
                     ngram_range,
                     min_df):
    """Constructs n-grams from text sequences and vectorizes them.
    
    Also scores the importance of the vectors and selects the most 
    important features.
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
