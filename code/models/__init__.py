import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional, BatchNormalization
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU

def build_embedding_layer(embedding_matrix, 
                          vocab_size, 
                          embedding_dim, 
                          max_sequence_length):
    # Input: Sequences of integers with input shape: (samples, indices)
    # Output: A 3D tensor of shape (samples, sequence_length, embedding_dim)
    #
    # Layer is frozen so that its weights (the embedding vectors)
    # will not be updated during training.
    #
    # Note: You can remove `weights` and `trainable` to train the embedding.
    return Embedding(vocab_size,
                     embedding_dim,
                     input_length=max_sequence_length,
                     weights=[embedding_matrix],
                     trainable=False)

def build_cnn_model(embedding_layer, max_sequence_length):
    input_layer = Input(shape=(max_sequence_length,), dtype='int32', name='input_layer')
    x = embedding_layer(input_layer)
    
    x = Flatten()(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)

    output_layer = Dense(3, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.summary() # Todo: Save to file
    return model
