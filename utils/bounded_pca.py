"""Float64 PCA using disk-backed features and components, without square diagonals."""
import numpy as np
from scipy.linalg import eigh


def disk_pca(features, num_pcs, block_size=256):
    if not isinstance(features, np.memmap) or features.dtype != np.float64:
        raise ValueError('disk_pca requires a float64 feature memmap')
    n, d = features.shape
    rank = min(n, d)
    if not 0 < num_pcs < rank or n < 2:
        raise ValueError('PCA dimension must be below min(samples, features)')
    mean = np.mean(features, axis=0)
    for start in range(0, n, block_size):
        part = features[start:start + block_size]
        if not np.isfinite(part).all():
            raise ValueError('Non-finite PCA features')
        part -= mean
    features.flush()
    dual = d > n
    # Fortran storage allows LAPACK to reuse the Gram matrix allocation.
    gram = np.empty((rank, rank), dtype=np.float64, order='F')
    for start in range(0, rank, block_size):
        stop = min(start + block_size, rank)
        if dual:
            gram[start:stop] = (features[start:stop] @ features.T) / (n - 1)
        else:
            gram[start:stop] = (features[:, start:stop].T @ features) / (n - 1)
    # Covariance is symmetric: use a real symmetric solver instead of ARPACK's
    # large nonsymmetric complex workspace. Same PCA objective and precision.
    eigenvalues, eigenvectors = eigh(gram, subset_by_index=(rank - num_pcs, rank - 1),
                                     overwrite_a=True, check_finite=False, driver='evr')
    del gram
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()
    output = np.memmap(str(features.filename) + '.components', mode='w+', dtype=np.float64,
                       shape=(d, num_pcs))
    if dual:
        eigenvectors *= (1 / np.sqrt(np.maximum(eigenvalues, 1e-9)) / np.sqrt(n - 1))
        for start in range(0, d, block_size):
            output[start:start + block_size] = features[:, start:start + block_size].T @ eigenvectors
    else:
        output[:] = eigenvectors
    output.flush()
    return output, eigenvalues, mean


def disk_whiten(components, eigenvalues, filename, block_size=256):
    output = np.memmap(filename, mode='w+', dtype=np.float64, shape=components.shape)
    scale = 1 / np.sqrt(eigenvalues + 1e-9)
    for start in range(0, len(components), block_size):
        output[start:start + block_size] = components[start:start + block_size] * scale
    output.flush()
    return output
