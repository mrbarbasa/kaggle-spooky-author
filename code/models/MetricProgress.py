from time import time

from keras.callbacks import Callback

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
