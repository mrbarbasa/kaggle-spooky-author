import numpy as np

def calculate_logloss(actual, predicted, epsilon=1e-15):
    """Calculate the multiclass logarithmic loss.

    Note that the natural logarithm is used.

    Parameters
    ----------
    actual : numpy.ndarray
        The actual class labels; has shape (num_samples,).
    predicted : numpy.ndarray
        The predicted class labels; has shape
        (num_samples, num_classes).
    epsilon : int, optional
        The minimum `epsilon` and maximum `1 - epsilon` probability
        values before taking the log (to avoid log extremes).

    Returns
    -------
    logloss : float
        The multiclass natural logarithmic loss.
    """

    prob_sum = 0.0
    clipped_preds = np.clip(predicted, epsilon, 1 - epsilon)

    for i, vector in enumerate(clipped_preds):
        true_class_idx = actual[i]
        true_class_prob = vector[true_class_idx]
        prob_sum += np.log(true_class_prob)
        
    num_samples = len(actual)
    logloss = -(prob_sum/num_samples)
    return logloss

def calculate_mean_logloss(private_logloss, public_logloss):
    n_test = 8392 # len(submission)
    n_private = n_test * 0.7
    n_public = n_test * 0.3
    mean_logloss = (private_logloss * n_private + public_logloss * n_public) / n_test
    print(f'Mean logloss: {mean_logloss:.5f}')
