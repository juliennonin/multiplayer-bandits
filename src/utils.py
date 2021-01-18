import numpy as np

def randmax(A):
    """Return an argmax at random in case of multiple maximizers
    @author: Émilie Kaufmann"""
    max_value = max(A)
    index = [i for i in range(len(A)) if A[i] == max_value]
    return np.random.choice(index)

def print_loading(i, n):
    m = int((100 * i / n) // 5)
    print(f"\r{i}/{n}", "[" + "="*m + " "*(20-m) + "]",f"({100 * i / n:.0f}%)", end='')