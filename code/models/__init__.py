from time import time
from datetime import timedelta
from contextlib import redirect_stdout

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional, BatchNormalization
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU

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
        with open(self.progress_file_path, 'a') as f:
            with redirect_stdout(f):
                print(f'----- Fold {self.nth_fold} of {self.n_splits} -----')

    def on_train_end(self, logs={}):
        fold_elapsed = round(time() - self.fold_start)
        fold_elapsed_str = str(timedelta(seconds=fold_elapsed))
        with open(self.progress_file_path, 'a') as f:
            with redirect_stdout(f):
                print(f'Fold {self.nth_fold} training runtime: '
                      f'{fold_elapsed_str}\n')

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = time()

    def on_epoch_end(self, epoch, logs={}):
        nth_epoch = epoch + 1
        score_improved = False
        epoch_elapsed = round(time() - self.epoch_start)
        epoch_elapsed_str = str(timedelta(seconds=epoch_elapsed))
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

        with open(self.progress_file_path, 'a') as f:
            with redirect_stdout(f):
                if score_improved:
                    print(f'Epoch {nth_epoch:05d}: {self.metric} improved '
                          f'from {self.best_score:.5f} to {current_score:.5f}; '
                          f'runtime {epoch_elapsed_str}; model saved')
                    self.best_score = current_score
                else:
                    print(f'Epoch {nth_epoch:05d}: {self.metric} did not improve '
                          f'from {self.best_score:.5f}; runtime {epoch_elapsed_str}')

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
    # and save only the best ones thus far
    checkpointer = ModelCheckpoint(model_file_path,
                                   monitor=monitored_metric,
                                   verbose=0,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode=mode,
                                   period=1)
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
    return [metric_progress, checkpointer, stopper, logger]

def save_model_summary(model, file_path):
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def build_cnn_model(embedding_layer, max_sequence_length):
    input_layer = Input(shape=(max_sequence_length,),
                        dtype='int32',
                        name='input_layer')
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
    return model

def build_gru_model(embedding_layer, max_sequence_length):
    input_layer = Input(shape=(max_sequence_length,),
                        dtype='int32',
                        name='input_layer')
    x = embedding_layer(input_layer)
    
    x = Flatten()(x)

    output_layer = Dense(3, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
