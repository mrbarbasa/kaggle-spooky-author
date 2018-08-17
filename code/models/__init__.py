import numpy as np

from time import time
from contextlib import redirect_stdout

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, SpatialDropout1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU

from utils import save_line_to_file, get_time_elapsed

class MetricProgress(Callback):
    def __init__(self, metric, mode, nth_fold, n_splits, progress_file_path):
        super(Callback, self).__init__()
        self.metric = metric
        self.mode = mode
        self.nth_fold = nth_fold
        self.n_splits = n_splits
        self.progress_file_path = progress_file_path
        self.best_score = float('inf')

    def on_train_begin(self, logs={}):
        self.fold_start = time()
        line = f'\n\n----- Fold {self.nth_fold} of {self.n_splits} -----'
        save_line_to_file(line, self.progress_file_path, 'a')

    def on_train_end(self, logs={}):
        _, fold_elapsed_str = get_time_elapsed(self.fold_start)
        line = f'Fold {self.nth_fold} training runtime: {fold_elapsed_str}'
        save_line_to_file(line, self.progress_file_path, 'a')

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = time()

    def on_epoch_end(self, epoch, logs={}):
        nth_epoch = epoch + 1
        score_improved = False
        line = None
        _, epoch_elapsed_str = get_time_elapsed(self.epoch_start)
        current_score = logs[self.metric]

        if self.mode == 'max':
            # If the current metric score is higher than the best one,
            # store the current one as the best
            score_improved = current_score > self.best_score
        else:
            # We assume that the mode is `min`
            # If the current metric score is lower than the best one,
            # store the current one as the best
            score_improved = current_score < self.best_score

        if score_improved:
            line = (f'Epoch {nth_epoch:03d}: {self.metric} improved '
                    f'from {self.best_score:.5f} to {current_score:.5f}; '
                    f'runtime {epoch_elapsed_str}; BEST YET')
            self.best_score = current_score
        else:
            line = (f'Epoch {nth_epoch:03d}: {self.metric} did not improve '
                    f'from {self.best_score:.5f}; runtime {epoch_elapsed_str}')
        save_line_to_file(line, self.progress_file_path, 'a')

def build_embedding_layer(embedding_matrix, 
                          vocab_size, 
                          embedding_dim, 
                          max_sequence_length):
    # Input: Sequences of integers with input shape: (samples, indices)
    # Output: A 3D tensor of shape (batch_size, sequence_length, embedding_dim)
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

def build_model_callbacks(monitored_metric,
                          mode,
                          progress_file_path,
                          model_file_path,
                          logger_file_path,
                          nth_fold,
                          n_splits):
    # Record model improvement progress based on the monitored metric
    metric_progress = MetricProgress(monitored_metric,
                                     mode, 
                                     nth_fold, 
                                     n_splits,
                                     progress_file_path)
    # Evaluate the best model at the end of every epoch
    # and save only the best one thus far
    # checkpointer = ModelCheckpoint(model_file_path,
    #                                monitor=monitored_metric,
    #                                verbose=0,
    #                                save_best_only=True,
    #                                save_weights_only=False,
    #                                mode=mode,
    #                                period=1)
    # Evaluate the metric at the end of every epoch
    # and stop training early if the metric has not
    # improved after `patience` epochs
    stopper = EarlyStopping(monitor=monitored_metric,
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            mode=mode,
                            baseline=None)
    # Log all the metrics at the end of every epoch
    logger = CSVLogger(logger_file_path, separator=',', append=False)
    # return [metric_progress, checkpointer, stopper, logger]
    return [metric_progress, stopper, logger]

def save_model_summary(model, file_path):
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def get_random_cnn_params(normal_arch_threshold=0.8):
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

def get_random_rnn_params(one_stack_threshold=0.7):
    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    use_gru_layer = bool(np.random.choice([True, False]))
    use_global_max_pooling_layer = bool(np.random.choice([True, False]))

    units = int(np.random.choice([32, 64, 128, 256, 300]))
    spatial_dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    use_extra_stack_value = float(np.random.uniform(0, 1))
    # `one_stack_threshold = 0.7` by default:
    # Use one RNN stack 70% of the time
    num_rnn_stacks = 2 if use_extra_stack_value > one_stack_threshold else 1

    return {
        'batch_size': batch_size,
        'use_gru_layer': use_gru_layer,
        'use_global_max_pooling_layer': use_global_max_pooling_layer,
        'units': units,
        'spatial_dropout_rate': spatial_dropout_rate,
        'optimizer': optimizer,
        'num_rnn_stacks': num_rnn_stacks,
    }

def get_random_mlp_params():
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
