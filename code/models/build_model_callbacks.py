# from keras.callbacks import ModelCheckpoint # Disabled for now
from keras.callbacks import EarlyStopping, CSVLogger

from .MetricProgress import MetricProgress

def build_model_callbacks(monitored_metric,
                          mode,
                          progress_file_path,
                          model_file_path,
                          logger_file_path,
                          nth_fold,
                          n_splits):
    """Build Keras model callbacks to help with monitoring and logging.

    Parameters
    ----------
    monitored_metric : string
        The monitored metric, such as 'val_loss' or 'val_acc'.
    mode : string
        The string 'max' or 'min'; this indicates if the monitored
        metric should maximize or minimize as a sign of improvement.
    progress_file_path : string
        A file path to where model improvement progress is recorded.
    model_file_path : string
        A file path to where the best model thus far is saved.
    logger_file_path : string
        A file path to where the metrics are recorded per epoch.
    nth_fold : int
        The current k-fold fold.
    n_splits : int
        The number of k-fold folds.

    Returns
    -------
    callbacks : list
        A list of callbacks to feed into the model during training.
    """

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
