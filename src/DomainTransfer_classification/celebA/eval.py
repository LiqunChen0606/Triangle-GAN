import numpy as np
def diff(x, y):
    shape = x.shape
    x = np.int32(x > 0.5*np.ones(shape))
    diff = np.int32(np.equal(x,y))
    diff = np.sum(diff, axis=1)
    diff = diff / 40.

    return np.mean(diff)
