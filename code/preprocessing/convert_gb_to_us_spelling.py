from .gb_to_us_dictionary import gb_to_us_dictionary

def convert_gb_to_us_spelling(word):
    # Return the American English spelling of the British-spelled word
    if word in gb_to_us_dictionary:
        return gb_to_us_dictionary[word]
    else:
        return word
