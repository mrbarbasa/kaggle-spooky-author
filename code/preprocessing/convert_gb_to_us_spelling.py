from .gb_to_us_dictionary import gb_to_us_dictionary

def convert_gb_to_us_spelling(word):
    """Convert a word from British English to American English spelling.

    Parameters
    ----------
    word : string
        The input word to convert, if found in the dictionary.
    
    Returns
    -------
    word : string
        The American English version of the input word, if found in the
        dictionary, or the unchanged input word if not found.
    """

    if word in gb_to_us_dictionary:
        return gb_to_us_dictionary[word]
    else:
        return word
