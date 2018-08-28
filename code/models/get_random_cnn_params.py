import numpy as np

def get_random_cnn_params(normal_arch_threshold=0.8):
    """Retrieve random CNN parameters and hyperparameters.

    Parameters
    ----------
    normal_arch_threshold : float, optional
        A fraction between 0 and 1 that specifies the probability of
        using the normal CNN architecture over the special architecture.
    
    Returns
    -------
    params : dict
        Model parameters and hyperparameters to govern the construction
        of a CNN model. They are:
        - batch_size : int
            The number of samples per batch; after a batch is trained,
            weights are updated.
        - filters : int
            The number of filters in a convolutional layer.
        - kernel_size : int
            The length of the 1D convolution window.
        - dropout_rate : float
            Fraction of the input units to drop.
        - optimizer : string
            An optimizer such as Adam or RMSProp.
        - use_special_arch : bool
            Whether or not to use the special CNN architecture.
        - normal_arch_params : dict
            This dictionary should only have keys if `use_special_arch`
            is False; otherwise, it is an empty dictionary.
            - num_conv_stacks : int
                The number of convolutional stacks.
            - add_extra_conv_layer : bool
                Add an extra convolutional layer whenever a convolutional
                layer appears.
            - add_dropout_layer : bool
                Add a dropout layer at the end of every convolutional
                stack, after the max pooling layer.
            - flatten : bool
                Whether or not to end the CNN model with a Keras Flatten
                and Dense layer, as opposed to one or two convolutional
                layers followed by a global max or average pooling layer.
            - use_global_max_pooling_layer : bool
                Only applies if `flatten` is False: End the model with a
                global max pooling layer instead of a global average.
            - add_final_dropout_layer : bool
                Add a final dropout layer right before the output layer.
            - pool_size : int
                Size of the max pooling windows.
            - final_dropout_rate : float
                Only applies if `add_final_dropout_layer` is True:
                Fraction of the input units to drop for the final
                dropout layer.
    """

    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    filters = int(np.random.choice([32, 64, 128, 256, 300]))
    kernel_size = int(np.random.choice([3, 5, 7, 9]))
    dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    special_arch_value = float(np.random.uniform(0, 1))
    # `normal_arch_threshold = 0.8` by default:
    # Use normal architecture 80% of the time
    use_special_arch = special_arch_value > normal_arch_threshold

    nap = {}

    if not use_special_arch:
        nap['num_conv_stacks'] = int(np.random.choice([1, 2, 3]))
        nap['add_extra_conv_layer'] = bool(np.random.choice([True, False]))
        nap['add_dropout_layer'] = bool(np.random.choice([True, False]))

        nap['flatten'] = bool(np.random.choice([True, False]))
        nap['use_global_max_pooling_layer'] = bool(np.random.choice([True, False]))
        nap['add_final_dropout_layer'] = bool(np.random.choice([True, False]))

        nap['pool_size'] = int(np.random.choice([2, 3, 4, 5]))
        nap['final_dropout_rate'] = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))

    return {
        'batch_size': batch_size,
        'filters': filters,
        'kernel_size': kernel_size,
        'dropout_rate': dropout_rate,
        'optimizer': optimizer,
        'use_special_arch': use_special_arch,
        'normal_arch_params': nap,
    }
