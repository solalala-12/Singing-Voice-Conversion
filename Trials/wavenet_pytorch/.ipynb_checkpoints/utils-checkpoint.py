import numpy as np


def debug_memory():
    # prints currently alive Tensors and Variables
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def quantize_signal(data, n_class):
    mu_x = np.sign(data) * np.log(1 + n_class * np.abs(data)) / np.log(n_class + 1)
    bins = np.linspace(-1, 1, n_class)
    quantized_data = np.digitize(mu_x, bins) - 1
    return quantized_data


def dequantize_signal(data, n_class):
    data = (data / n_class) * 2. - 1
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(n_class + 1)) - 1) / n_class
    return s
