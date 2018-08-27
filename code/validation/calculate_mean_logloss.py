def calculate_mean_logloss(private_logloss, public_logloss):
    n_test = 8392 # len(submission)
    n_private = n_test * 0.7
    n_public = n_test * 0.3
    mean_logloss = (private_logloss * n_private + public_logloss * n_public) / n_test
    print(f'Mean logloss: {mean_logloss:.5f}')
    
