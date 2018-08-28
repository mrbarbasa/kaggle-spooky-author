import numpy as np

from tqdm import tqdm

def load_embeddings(embeddings_path):
    """Load the GloVe or fastText embeddings into a dictionary. This maps
    words (as strings) to their vector representations (as float vectors).

    The first line of a fastText embeddings file contains:
    - The number of words in the vocabulary.
    - The size of each vector.

    Otherwise, the rest of a fastText file has the same format as GloVe:
    - Each line contains a string word followed by its vector.
    - Each value is space-separated.
    - Words are listed by descending frequency.

    Parameters
    ----------
    embeddings_path : string
        The file path to the GloVe or fastText embeddings.

    Returns
    -------
    embeddings_index : dict
        The embeddings index, which maps a word to its vector
        representation.
    """

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
