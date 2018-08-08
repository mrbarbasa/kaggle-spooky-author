import matplotlib.pyplot as plt

def display_metric_vs_epochs_plot(scores, metric, fold):
    metric_fold_scores = scores[metric][fold]
    epochs = range(1, len(metric_fold_scores) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, metric_fold_scores, 'bo', label=f'Training {metric}')
    plt.plot(epochs, scores[f'val_{metric}'][fold], 'b', label=f'Validation {metric}')
    plt.title(f'Training and validation {metric} for fold index {fold}')
    plt.legend()
    plt.ylabel(metric.title())
    plt.xlabel('Number of Epochs')
    plt.show()
