from contextlib import redirect_stdout

def save_model_summary(model, file_path):
    """Save a Keras model summary to file.

    Parameters
    ----------
    model : keras.models.Model
        A compiled Keras model.
    file_path : string
        The output file path.

    Returns
    -------
    None
    """

    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
