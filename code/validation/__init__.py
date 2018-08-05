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
    for i, vector in enumerate(predicted):
        class_idx = actual[i]
        prob = max(min(vector[class_idx],1-epsilon), epsilon)
        prob_sum += np.log(prob)
    num_samples = len(actual)
    logloss = -(prob_sum/num_samples)
    return logloss
