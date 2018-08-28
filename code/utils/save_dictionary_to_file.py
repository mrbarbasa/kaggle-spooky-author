import json

def save_dictionary_to_file(dictionary, file_path):
    """Save a dictionary to a JSON file.

    Parameters
    ----------
    dictionary : dict
        The dictionary to save to a JSON file.
    file_path : string
        The JSON file path to save the dictionary to.

    Returns
    -------
    None
    """

    with open(file_path, 'w') as f:
        json.dump(dictionary, f, indent=4)
