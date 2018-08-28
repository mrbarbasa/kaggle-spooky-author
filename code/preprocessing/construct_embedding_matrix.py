import numpy as np

def construct_embedding_matrix(word_index,
                               embeddings_index,
                               embedding_dim,
                               max_features):
    """Construct the embedding weight matrix.

    Parameters
    ----------
    word_index : dict
        The computed word index of the training set words, which maps a
        word to its index.
    embeddings_index : dict
        The pre-trained embeddings index, which maps a word to its vector
        representation.
    embedding_dim : int
        The dimensionality of the embedding space.
    max_features : int
        The maximum number of features (vocabulary words) to include in
        every sequence; the larger (less common) word indices get
        omitted first. If this is None, then no words get omitted.
    
    Returns
    -------
    results : tuple
        - embedding_matrix : numpy.ndarray
            A 2D matrix of weights in the shape of
            (vocab_size, embedding_dim).
        - vocab_size : int
            The size of the vocabulary.
        - num_unknown : int
            The number of unknown words (or words not found) in the
            pre-trained embeddings.
    """

    # Start with the actual vocabulary size
    vocab_size = len(word_index)

    # Then limit the vocabulary to the maximum number of features.
    # Note that `max_features` may be None, so first check if it exists.
    vocab_size = min(max_features, vocab_size) if max_features else vocab_size
    
    # We need to add 1 here because Keras has a quirk that it will
    # discard the very last (highest index and least common) feature
    # if we use the exact vocabulary size
    vocab_size += 1

    # Initialize the embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Keep track of the number of words not found in the pre-trained embeddings
    num_unknown = 0

    # Loop over each of the first `vocab_size` words of the `word_index`
    # built from the dataset and retrieve its embedding vector from the
    # pre-trained `embeddings_index`
    for word, i in word_index.items():
        # The word index may be larger than the vocab size, due to
        # https://github.com/keras-team/keras/issues/7551 (also see
        # the function `preprocessing.compute_word_index` for details),
        # so move on to the next word index if so (word indices may not
        # be in order)
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        # A word in our corpus has been found in the pre-trained embeddings
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # Words not found in the `embeddings_index` will have their
            # vectors in `embedding_matrix` remain as all zeros
            num_unknown += 1
    return embedding_matrix, vocab_size, num_unknown
