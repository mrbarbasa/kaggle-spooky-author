from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout

# Based on Google's `mlp_model` architecture:
# https://developers.google.com/machine-learning/guides/text-classification/step-4
#
# Note that Google appears to define the "number of layers" in an MLP
# to include the output layer. So let's call it `num_total_layers` to
# avoid confusion.
def build_mlp_model(input_shape, params):
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
