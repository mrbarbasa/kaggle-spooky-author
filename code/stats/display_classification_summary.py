import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

def display_classification_summary(y_valid, y_pred, labels, target_names):
    """Display a classification report and confusion matrix.

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
    
    Returns
    -------
    None
    """

    # These label variables are like `y_train_integers` in format
    y_valid_class_labels = np.argmax(y_valid, axis=1)
    y_pred_class_labels = np.argmax(y_pred, axis=1)
    report = classification_report(y_valid_class_labels,
                                   y_pred_class_labels,
                                   labels=labels,
                                   target_names=target_names)
    conf_matrix = confusion_matrix(y_valid_class_labels,
                                   y_pred_class_labels,
                                   labels=labels)
    label_1 = target_names[0]
    label_2 = target_names[1]
    label_3 = target_names[2]
    print('\n            ----- Classification Report -----')
    print(report)
    print('            ----- Confusion Matrix -----')
    print(f'True Labels  {label_1}  {conf_matrix[0]}')
    print(f'             {label_2}  {conf_matrix[1]}')
    print(f'             {label_3}  {conf_matrix[2]}')
    print(f'                    {label_1}  {label_2}  {label_3}')
    print('                  Predicted Labels')
