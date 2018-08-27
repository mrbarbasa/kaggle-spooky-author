import numpy as np

def get_random_rnn_params(one_stack_threshold=0.7):
    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    use_gru_layer = bool(np.random.choice([True, False]))
    use_global_max_pooling_layer = bool(np.random.choice([True, False]))

    units = int(np.random.choice([32, 64, 128, 256, 300]))
    spatial_dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    use_extra_stack_value = float(np.random.uniform(0, 1))
    # `one_stack_threshold = 0.7` by default:
    # Use one RNN stack 70% of the time
    num_rnn_stacks = 2 if use_extra_stack_value > one_stack_threshold else 1

    return {
        'batch_size': batch_size,
        'use_gru_layer': use_gru_layer,
        'use_global_max_pooling_layer': use_global_max_pooling_layer,
        'units': units,
        'spatial_dropout_rate': spatial_dropout_rate,
        'optimizer': optimizer,
        'num_rnn_stacks': num_rnn_stacks,
    }
