from random import randint
import torch


def synthetic_data_gen():
    sequences = 10000
    t = 50

    data = torch.rand(sequences, t, 1)

    for j in range(sequences):
        i = randint(19, 29)
        for k in range(i - 5, i + 5):
            data[j, k] = data[j, k] * 0.1

    return data