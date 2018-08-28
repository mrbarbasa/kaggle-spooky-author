def calculate_mean_logloss(private_logloss, public_logloss):
    """Calculate the mean logloss of the Kaggle submission.

    After submitting to Kaggle, we can calculate the mean logloss across
    the entire test dataset as follows:

    Given:
        n_test = 8392
        %_private = 0.7
        %_public = 0.3
        private_logloss # Retrieve from Kaggle after submission
        public_logloss # Retrieve from Kaggle after submission
        
    Mean logloss = (private_logloss * n_private
                    + public_logloss * n_public)
                   / n_test
                 = (private_logloss * (%_private * n_test)
                    + public_logloss * (%_public * n_test))
                   / n_test
                   
    Where n_test = n_private + n_public

    Parameters
    ----------
    private_logloss : float
        The private leaderboard logloss score retrieved from Kaggle.
    public_logloss : float
        The public leaderboard logloss score retrieved from Kaggle.

    Returns
    -------
    None
    """

    n_test = 8392 # len(submission)
    n_private = n_test * 0.7
    n_public = n_test * 0.3
    mean_logloss = (private_logloss * n_private + public_logloss * n_public) / n_test
    print(f'Mean logloss: {mean_logloss:.5f}')
