from contextlib import redirect_stdout

from .display_classification_summary import display_classification_summary

def save_classification_summary(y_valid,
                                y_pred,
                                labels,
                                target_names,
                                file_path,
                                mode='w'):
    with open(file_path, mode) as f:
        with redirect_stdout(f):
            display_classification_summary(y_valid,
                                           y_pred,
                                           labels,
                                           target_names)
