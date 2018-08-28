import pandas as pd

def load_data(train_path, test_path, submission_path):
    """Load data from train, test, and sample submission CSV files into
    pandas DataFrames.

    Parameters
    ----------    
    train_path : string
        The CSV file path to the train set.
    test_path : string
        The CSV file path to the test set.
    submission_path : string
        The CSV file path to the sample Kaggle submission.

    Returns
    -------
    results : tuple
        - train : pandas.DataFrame
            The train set.
        - test : pandas.DataFrame
            The test set.
        - submission : pandas.DataFrame
            A sample Kaggle submission.
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)
    return train, test, submission
