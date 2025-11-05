import jax.numpy as jnp

def rbf_kernel(x, y, bandwidth=1.0):
    """RBF kernel k(x, y) = exp(-||x - y||^2 / (2 * h^2))"""
    x_norm = jnp.sum(x**2, axis=1, keepdims=True)
    y_norm = jnp.sum(y**2, axis=1, keepdims=True)
    sq_dists = x_norm + y_norm.T - 2 * x @ y.T
    return jnp.exp(-0.5 * sq_dists / (bandwidth**2))

def compute_mmd2(x, y, bandwidth=1.0):
    """Compute unbiased squared MMD between two sets of samples x, y."""
    Kxx = rbf_kernel(x, x, bandwidth)
    Kyy = rbf_kernel(y, y, bandwidth)
    Kxy = rbf_kernel(x, y, bandwidth)

    n = x.shape[0]
    m = y.shape[0]

    sum_Kxx = jnp.sum(Kxx) / (n * n)
    sum_Kyy = jnp.sum(Kyy) / (m * m)
    sum_Kxy = jnp.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2 * sum_Kxy
    return mmd2
