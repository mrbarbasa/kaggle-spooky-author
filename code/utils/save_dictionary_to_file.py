import json

def save_dictionary_to_file(dictionary, file_path):
    with open(file_path, 'w') as f:
        json.dump(dictionary, f, indent=4)
