print("Importing modules...")
import numpy as np
np.random.seed(462019)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

print("Loading data...")
loaded = np.load("zebrafish_all_old.npz")
X, y = loaded['inputs'], loaded['targets']
print("Shuffling data...")
X_shuffled, y_shuffled = unison_shuffled_copies(X, y)
print("Saving data...")
np.savez_compressed("zebrafish_all.npz", inputs=X_shuffled, targets=y_shuffled)
print("Done")