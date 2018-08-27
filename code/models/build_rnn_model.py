from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, SpatialDropout1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU

def build_rnn_model(embedding_layer, max_sequence_length, params):
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
