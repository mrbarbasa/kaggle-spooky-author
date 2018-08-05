import pandas as pd

def load_data():
    INPUT_DIR = '../input/'
    TRAIN_FILE_PATH = INPUT_DIR + 'train.csv'
    TEST_FILE_PATH = INPUT_DIR + 'test.csv'
    SAMPLE_SUBMISSION_FILE_PATH = INPUT_DIR + 'sample_submission.csv'

    train = pd.read_csv(TRAIN_FILE_PATH)
    test = pd.read_csv(TEST_FILE_PATH)
    submission = pd.read_csv(SAMPLE_SUBMISSION_FILE_PATH)

    return train, test, submission
