import string
import numpy as np
from tqdm import tqdm

from nltk import pos_tag
from nltk.text import Text
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from preprocessing.gb_to_us_dictionary import gb_to_us_dictionary

nltk_stopwords = set(stopwords.words('english'))

def convert_gb_to_us_spelling(word):
    # Return the American English spelling of the British-spelled word
    if word in gb_to_us_dictionary:
        return gb_to_us_dictionary[word]
    else:
        return word

def get_wordnet_pos(tag):
    # `tag` is the Penn Treebank tag that NLTK
    # outputs for its `pos_tag` function
    if tag.startswith('V'):
        return wordnet.VERB # v
    elif tag.startswith('J'):
        return wordnet.ADJ # a
    elif tag.startswith('R'):
        return wordnet.ADV # r
    else: # Default or starts with 'N'
        return wordnet.NOUN # n

def process_text(text,
                 lower=True,
                 remove_punc=False,
                 normalize_spelling=False,
                 stem=False,
                 lemmatize=False,
                 remove_stopwords=False):
    # Convert to lowercase in order to treat "the" and "The" as the same word
    if lower:
        text = text.lower()
    
    # Remove punctuation
    if remove_punc:
        regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex_punc.sub('', text)
        
    # Tokenize words, treating punctuation (if any) as separate tokens
    tokens = word_tokenize(text)
    
    # Convert British English to American English spelling
    if normalize_spelling:
        tokens = [convert_gb_to_us_spelling(w) for w in tokens]
    
    # Reduce words to their stem
    if stem:
        porter = PorterStemmer()
        tokens = [porter.stem(w) for w in tokens]
    
    # Lemmatize words
    if lemmatize:        
        lem = WordNetLemmatizer()
        word_tags = pos_tag(tokens)
        tokens = [lem.lemmatize(w, pos=get_wordnet_pos(t)) for w,t in word_tags]
        
    # Remove stopwords
    if remove_stopwords:
        tokens = [w for w in tokens if not w in nltk_stopwords]
        
    return ' '.join(tokens)

def compute_word_index(X_train_sequences,
                       X_test_sequences,
                       max_features,
                       max_sequence_length):
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
    # Note that despite the `num_words` set in the Tokenizer, all unique
    # tokens are stored here as if `num_words` were set to `None`.
    # https://github.com/keras-team/keras/issues/7551
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
