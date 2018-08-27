from nltk.corpus import wordnet

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
