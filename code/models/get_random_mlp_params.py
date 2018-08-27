import numpy as np

def get_random_mlp_params():
    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    units = int(np.random.choice([32, 64, 128, 256, 300]))
    dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    num_total_layers = int(np.random.choice([2, 3]))

    return {
        'batch_size': batch_size,
        'units': units,
        'dropout_rate': dropout_rate,
        'optimizer': optimizer,
        'num_total_layers': num_total_layers,
    }
