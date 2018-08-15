import matplotlib.pyplot as plt

def display_metric_vs_epochs_plot(scores, metric, nth_iter, nth_fold):
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
