import json
import pandas as pd

from time import time
from datetime import timedelta
from contextlib import redirect_stdout

def load_data(train_path, test_path, submission_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)
    return train, test, submission

def save_line_to_file(line, file_path, mode):
    with open(file_path, mode) as f:
        with redirect_stdout(f):
            print(line)

def format_time_str(num_seconds):
    return str(timedelta(seconds=round(num_seconds)))

def get_time_elapsed(event_start):
    event_elapsed = time() - event_start
    event_elapsed_str = format_time_str(event_elapsed)
    return event_elapsed, event_elapsed_str

def save_dictionary_to_file(dictionary, file_path):
    with open(file_path, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_dictionary_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
