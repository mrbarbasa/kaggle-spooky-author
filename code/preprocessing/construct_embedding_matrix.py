import numpy as np

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
