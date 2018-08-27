from keras.layers import Embedding

def build_embedding_layer(embedding_matrix, 
                          vocab_size, 
                          embedding_dim, 
                          max_sequence_length):
    # Input: Sequences of integers with input shape: (samples, indices)
    # Output: A 3D tensor of shape (batch_size, sequence_length, embedding_dim)
    #
    # Layer is frozen so that its weights (the embedding vectors)
    # will not be updated during training.
    #
    # Note: You can remove `weights` and `trainable` to train the embedding.
    return Embedding(vocab_size,
                     embedding_dim,
                     input_length=max_sequence_length,
                     weights=[embedding_matrix],
                     trainable=False)
