import pandas as pd

def load_data(train_path, test_path, submission_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)
    return train, test, submission
