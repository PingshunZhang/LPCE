import numpy as np

def build_graphs_Full3(n: int, n_pad: int = None, edge_pad: int = None):
    A = np.ones((n, n))
    edge_num = int(np.sum(A, axis=(0, 1)))
    idxs = np.nonzero(A)

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    return np.array(idxs, dtype=np.int64)