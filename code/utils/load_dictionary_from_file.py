import json

def load_dictionary_from_file(file_path):
    """Load a dictionary from a JSON file.

    Parameters
    ----------
    file_path : string
        The JSON file path to load the dictionary from.

    Returns
    -------
    dictionary : dict
        The dictionary loaded from a JSON file.
    """

    with open(file_path, 'r') as f:
        return json.load(f)
