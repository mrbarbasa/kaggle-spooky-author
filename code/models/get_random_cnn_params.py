import numpy as np

def get_random_cnn_params(normal_arch_threshold=0.8):
    batch_size = int(np.random.choice([32, 64, 128, 256, 512]))

    filters = int(np.random.choice([32, 64, 128, 256, 300]))
    kernel_size = int(np.random.choice([3, 5, 7, 9]))
    dropout_rate = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    optimizer = str(np.random.choice(['adam', 'rmsprop']))

    special_arch_value = float(np.random.uniform(0, 1))
    # `normal_arch_threshold = 0.8` by default:
    # Use normal architecture 80% of the time
    use_special_arch = special_arch_value > normal_arch_threshold

    nap = {}

    if not use_special_arch:
        nap['num_conv_stacks'] = int(np.random.choice([1, 2, 3]))
        nap['add_extra_conv_layer'] = bool(np.random.choice([True, False]))
        nap['add_dropout_layer'] = bool(np.random.choice([True, False]))

        nap['flatten'] = bool(np.random.choice([True, False]))
        nap['use_global_max_pooling_layer'] = bool(np.random.choice([True, False]))
        nap['add_final_dropout_layer'] = bool(np.random.choice([True, False]))

        nap['pool_size'] = int(np.random.choice([2, 3, 4, 5]))
        nap['final_dropout_rate'] = float(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))

    return {
        'batch_size': batch_size,
        'filters': filters,
        'kernel_size': kernel_size,
        'dropout_rate': dropout_rate,
        'optimizer': optimizer,
        'use_special_arch': use_special_arch,
        'normal_arch_params': nap,
    }
