import random
import numpy as np


def addition_dataset(memory_size, length=10000):
    X = []
    Y = []
    for _ in range(length):
        x = random.randint(0, memory_size-1)
        y = random.randint(0, memory_size-1)
        inputs = [0 for i in range(memory_size)]
        inputs[0] = x
        inputs[1] = y
        outputs = [memory_size for i in range(memory_size)]
        outputs[0] = (x + y) % memory_size
        X.append(inputs)
        Y.append(outputs)

    return np.array(X), np.array(Y)


def brz_dataset(memory_size, length=10000):
    """1 if first input value is 0 else 2"""
    X = []
    Y = []
    for _ in range(length):
        x = random.randint(0, memory_size-1)
        inputs = [0 for i in range(memory_size)]
        inputs[0] = x
        outputs = [memory_size for i in range(memory_size)]
        outputs[0] = 1 if x == 0 else 2
        X.append(inputs)
        Y.append(outputs)

    return np.array(X), np.array(Y)


def sort_dataset(memory_size, length=10000, length_min=2, length_max=2):
    """1 if first input value is 0 else 2"""
    X = []
    Y = []
    for _ in range(length):
        length = random.randint(length_min, length_max)
        arr = [0 for i in range(memory_size)]
        for i in range(length):
            arr[i] = random.randint(1, memory_size//2-1)
        X.append(arr)
        y = sorted(arr[0:length]) + \
            [memory_size for _ in range(memory_size-length)]
        Y.append(y)

    return np.array(X), np.array(Y)
