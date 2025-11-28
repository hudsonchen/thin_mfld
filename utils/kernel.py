import jax.numpy as jnp
import jax

def gaussian_kernel(x, y, bandwidth):
    """RBF kernel k(x, y) = exp(-||x - y||^2 / (2 * h^2))
    x: (d,)
    y: (d,)
    bandwidth: scalar
    returns: scalar
    """
    sq_dist = jnp.sum((x - y) ** 2)
    return jnp.exp(-0.5 * sq_dist / (bandwidth ** 2))

def compute_mmd2(x, y, bandwidth=1.0):
    """Compute unbiased squared MMD between two sets of samples x, y."""
    gaussian_kernel_vmap = jax.vmap(jax.vmap(gaussian_kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))
    Kxx = gaussian_kernel_vmap(x, x, bandwidth)
    Kyy = gaussian_kernel_vmap(y, y, bandwidth)
    Kxy = gaussian_kernel_vmap(x, y, bandwidth)

    n = x.shape[0]
    m = y.shape[0]

    sum_Kxx = jnp.sum(Kxx) / (n * n)
    sum_Kyy = jnp.sum(Kyy) / (m * m)
    sum_Kxy = jnp.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2 * sum_Kxy
    return mmd2
