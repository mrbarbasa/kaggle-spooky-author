from keras.layers import Embedding

def build_embedding_layer(embedding_matrix, 
                          vocab_size, 
                          embedding_dim, 
                          max_sequence_length):
    """Build an embedding layer using Keras.

    The layer is frozen so that its weights (the embedding vectors)
    will not be updated during training.

    Note: The parameters `weights` and `trainable` can be removed
    in order to train the embedding.

    Parameters
    ----------
    embedding_matrix : numpy.ndarray
        A 2D matrix of weights in the shape of (vocab_size, embedding_dim).
    vocab_size : int
        The size of the vocabulary.
    embedding_dim : int
        The dimensionality of the embedding space.
    max_sequence_length : int
        The maximum length of a sequence (sentence).

    Returns
    -------
    model : keras.layers.Embedding
        A Keras Embedding layer that outputs a 3D tensor of shape
        (batch_size, max_sequence_length, embedding_dim).
    """

    return Embedding(vocab_size,
                     embedding_dim,
                     input_length=max_sequence_length,
                     weights=[embedding_matrix],
                     trainable=False)
