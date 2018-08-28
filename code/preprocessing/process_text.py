import string

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from .convert_gb_to_us_spelling import convert_gb_to_us_spelling
from .get_wordnet_pos import get_wordnet_pos

nltk_stopwords = set(stopwords.words('english'))

def process_text(text,
                 lower=True,
                 remove_punc=False,
                 normalize_spelling=False,
                 stem=False,
                 lemmatize=False,
                 remove_stopwords=False):
    """Apply text preprocessing to the input text.

    Parameters
    ----------
    text : string
        A sequence (sentence) of words.
    lower : bool, optional
        Whether or not to lowercase the text.
    remove_punc : bool, optional
        Whether or not to remove all punctuation.
    normalize_spelling : bool, optional
        Whether or not to convert British English to American English
        spelling.
    stem : bool, optional
        Whether or not to stem words.
    lemmatize : bool, optional
        Whether or not to lemmatize words.
    remove_stopwords : bool, optional
        Whether or not to remove stopwords.

    Returns
    -------
    text : string
        The preprocessed sequence.
    """

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
