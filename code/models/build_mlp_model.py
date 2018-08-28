from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout

def build_mlp_model(input_shape, params):
    """Build an MLP model using Keras.

    Based on Google's `mlp_model` architecture:
    https://developers.google.com/machine-learning/guides/text-classification/step-4.

    Note that Google appears to define the "number of layers" in an MLP
    to include the output layer. So let's call it `num_total_layers` to
    avoid confusion.

    Parameters
    ----------
    input_shape : tuple
        A tuple in the form of (num_features,).
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

    Returns
    -------
    model : keras.models.Model
        The compiled Keras model.
    """

    K.clear_session()

    input_layer = Input(shape=input_shape,
                        dtype='float',
                        name='input_layer')

    units = params['units']
    dropout_rate = params['dropout_rate']
    optimizer = params['optimizer']
    num_total_layers = params['num_total_layers']

    x = Dropout(dropout_rate)(input_layer)

    # We subtract 1 because the number of total layers includes the
    # output layer
    for i in range(num_total_layers-1):
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    output_layer = Dense(3, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
