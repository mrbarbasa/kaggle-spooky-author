from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, SpatialDropout1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU

def build_rnn_model(embedding_layer, max_sequence_length, params):
    """Build an RNN model using Keras.

    Parameters
    ----------
    embedding_layer : keras.layers.Embedding
        A preconstructed embedding layer.
    max_sequence_length : int
        The maximum length of a sequence (sentence).
    params : dict
        Model parameters and hyperparameters to govern the construction
        of the RNN model. They are:
        - batch_size : int
            The number of samples per batch; after a batch is trained,
            weights are updated.
        - use_gru_layer : bool
            Whether or not to use GRU layers; otherwise, use LSTM.
        - use_global_max_pooling_layer : bool
            Whether or not to end the model with just a global max
            pooling layer; otherwise, use a concatenated global average
            and global max pooling layer.
        - units : int
            The number of hidden units in a GRU or LSTM layer.
        - spatial_dropout_rate : float
            Fraction of the input units to drop. Drops entire 1D feature
            maps instead of individual elements.
        - optimizer : string
            An optimizer such as Adam or RMSProp.
        - num_rnn_stacks : int
            The number of GRU or LSTM stacks.

    Returns
    -------
    model : keras.models.Model
        The compiled Keras model.
    """

    K.clear_session()

    input_layer = Input(shape=(max_sequence_length,),
                        dtype='int32',
                        name='input_layer')
    x = embedding_layer(input_layer)
    
    RNNLayer = CuDNNGRU if params['use_gru_layer'] else CuDNNLSTM
    use_global_max_pooling_layer = params['use_global_max_pooling_layer']
    units = params['units']
    spatial_dropout_rate = params['spatial_dropout_rate']
    optimizer = params['optimizer']
    num_rnn_stacks = params['num_rnn_stacks']
    
    for i in range(num_rnn_stacks):
        x = SpatialDropout1D(spatial_dropout_rate)(x)
        x = Bidirectional(RNNLayer(units, return_sequences=True))(x)

    if use_global_max_pooling_layer:
        x = GlobalMaxPooling1D()(x)
    else:
        avg_pooling = GlobalAveragePooling1D()(x)
        max_pooling = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pooling, max_pooling])

    output_layer = Dense(3, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
