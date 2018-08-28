from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    """Retrieve the WordNet part-of-speech (POS) tag for the given Penn
    Treebank tag.

    Parameters
    ----------
    tag : string
        The Penn Treebank tag (that NLTK outputs for its `pos_tag`
        function).
    
    Returns
    -------
    wordnet_tag : string
        The desired WordNet tag equivalent to the Penn Treebank tag.
    """

    if tag.startswith('V'):
        return wordnet.VERB # v
    elif tag.startswith('J'):
        return wordnet.ADJ # a
    elif tag.startswith('R'):
        return wordnet.ADV # r
    else: # Default or starts with 'N'
        return wordnet.NOUN # n
