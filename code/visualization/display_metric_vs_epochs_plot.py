import matplotlib.pyplot as plt

def display_metric_vs_epochs_plot(scores, metric, nth_iter, nth_fold):
    """Display a metric vs. epochs plot.

    Both the training and validation scores will be plotted for the
    chosen metric.

    Parameters
    ----------
    scores : pandas.DataFrame
        Scores containing the `epoch` column and metric columns such as
        `acc`, `loss`, `val_acc`, and `val_loss`.
    metric : string
        The metric to display, such as 'loss' or 'acc' (note that `val_`
        is also appended automatically and should not be provided).
    nth_iter : int
        The current random search iteration.
    nth_fold : int
        The current k-fold fold.

    Returns
    -------
    None
    """

    metric_fold_scores = scores[metric].values.tolist()
    epochs = range(1, len(metric_fold_scores) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, metric_fold_scores, 'bo', label=f'Training {metric}')
    plt.plot(epochs,
             scores[f'val_{metric}'].values.tolist(),
             'b',
             label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric.title()} for '
              f'Iteration {nth_iter} Fold {nth_fold}')
    plt.legend()
    plt.ylabel(metric.title())
    plt.xlabel('Number of Epochs')
    plt.show()
