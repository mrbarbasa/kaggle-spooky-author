import numpy as np

def get_random_mlp_params():
    """Retrieve random MLP parameters and hyperparameters.

    Based on Google's `mlp_model` architecture:
    https://developers.google.com/machine-learning/guides/text-classification/step-4.

    Note that Google appears to define the "number of layers" in an MLP
    to include the output layer. So let's call it `num_total_layers` to
    avoid confusion.

    Parameters
    ----------
    None

    Returns
    -------
    params : dict
        Model parameters and hyperparameters to govern the construction
        of the MLP model. They are:
        - batch_size : int
            The number of samples per batch; after a batch is trained,
            weights are updated.
        - units : int
            The number of hidden units in a fully connected layer.
        - dropout_rate : float
            Fraction of the input units to drop.
        - optimizer : string
            An optimizer such as Adam or RMSProp.
        - num_total_layers : int
            The number of total fully connected layers (including the
            output layer).
    """

    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    units = int(np.random.choice([32, 64, 128, 256, 300]))
    dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    num_total_layers = int(np.random.choice([2, 3]))

    return {
        'batch_size': batch_size,
        'units': units,
        'dropout_rate': dropout_rate,
        'optimizer': optimizer,
        'num_total_layers': num_total_layers,
    }
