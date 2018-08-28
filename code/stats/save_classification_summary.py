from contextlib import redirect_stdout

from .display_classification_summary import display_classification_summary

def save_classification_summary(y_valid,
                                y_pred,
                                labels,
                                target_names,
                                file_path,
                                mode='w'):
    """Save a classification report and confusion matrix to file.

    Parameters
    ----------
    y_valid : numpy.ndarray
        Actual labels for the validation data.
    y_pred : numpy.ndarray
        Model predictions.
    labels : list
        A list of integer index labels.
    target_names : list
        A list of target class strings mapped to the `labels`.
    file_path : string
        The output file path to save the summary to.
    mode : string, optional
        Whether to write 'w' or append 'a' to the output file.
    
    Returns
    -------
    None
    """

    with open(file_path, mode) as f:
        with redirect_stdout(f):
            display_classification_summary(y_valid,
                                           y_pred,
                                           labels,
                                           target_names)
