from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

def build_cnn_model(embedding_layer, max_sequence_length, params):
    K.clear_session()

    input_layer = Input(shape=(max_sequence_length,),
                        dtype='int32',
                        name='input_layer')
    x = embedding_layer(input_layer)

    filters = params['filters']
    kernel_size = params['kernel_size']
    dropout_rate = params['dropout_rate']
    optimizer = params['optimizer']
    use_special_arch = params['use_special_arch']

    if use_special_arch:
        x = Dropout(dropout_rate)(x)
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(filters, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    else:
        nap = params['normal_arch_params']
        num_conv_stacks = nap['num_conv_stacks']
        add_extra_conv_layer = nap['add_extra_conv_layer']
        add_dropout_layer = nap['add_dropout_layer']
        flatten = nap['flatten']
        use_global_max_pooling_layer = nap['use_global_max_pooling_layer']
        add_final_dropout_layer = nap['add_final_dropout_layer']
        pool_size = nap['pool_size']
        final_dropout_rate = nap['final_dropout_rate']

        for i in range(num_conv_stacks):
            x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
            if add_extra_conv_layer:
                x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)

            x = MaxPooling1D(pool_size, padding='same')(x)
            if add_dropout_layer:
                x = Dropout(dropout_rate)(x)
            
        if flatten:
            x = Flatten()(x)
            x = Dense(filters, activation='relu')(x)
        else:
            x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
            if add_extra_conv_layer:
                x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)

            if use_global_max_pooling_layer:
                x = GlobalMaxPooling1D()(x)
            else:
                x = GlobalAveragePooling1D()(x)

        if add_final_dropout_layer:
            x = Dropout(final_dropout_rate)(x)

    output_layer = Dense(3, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
