import json

def load_dictionary_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
