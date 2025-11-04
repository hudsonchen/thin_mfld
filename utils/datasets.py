# pip install scikit-learn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import jax
import jax.numpy as jnp

def load_boston(test_size=0.2, seed=42, standardize_X=True, standardize_y=False):
    """
    Returns JAX arrays and (optionally) standardization stats.
    Shapes:
      X_*: (N, 13), y_*: (N,)  # regression target
    """
    # Load from OpenML (since sklearn.load_boston is deprecated)
    ds = fetch_openml(name="boston", version=1, as_frame=True)
    X = ds.data.to_numpy(dtype=np.float32)
    y = ds.target.to_numpy().astype(np.float32).reshape(-1, 1)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Standardize using train stats
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    if standardize_X:
        X_tr = (X_tr - x_mean) / x_std
        X_te = (X_te - x_mean) / x_std

    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std  = y_tr.std(axis=0, keepdims=True) + 1e-8
    if standardize_y:
        y_tr = (y_tr - y_mean) / y_std
        y_te = (y_te - y_mean) / y_std

    # Convert to JAX arrays (targets flattened to shape (N,))
    out = {
        "Z": jnp.asarray(X_tr),
        "y": jnp.asarray(y_tr.squeeze(-1)),
        "Z_test":  jnp.asarray(X_te),
        "y_test":  jnp.asarray(y_te.squeeze(-1)),
        "z_stats": (jnp.asarray(x_mean.squeeze(0)), jnp.asarray(x_std.squeeze(0))),
        "y_stats": (jnp.asarray(y_mean.item()),     jnp.asarray(y_std.item())),
    }
    return out

def batch_iter(X, y, batch_size, rng):
    """
    JAX-friendly shuffling batch iterator.
    rng: jax.random.PRNGKey
    """
    n = X.shape[0]
    idx = jax.random.permutation(rng, n)
    for start in range(0, n, batch_size):
        sl = idx[start:start + batch_size]
        yield X[sl], y[sl]

# ---- Example usage ----
# data = load_boston_jax(standardize_X=True, standardize_y=False)
# rng = jax.random.PRNGKey(0)
# for xb, yb in batch_iter(data["X_train"], data["y_train"], batch_size=64, rng=rng):
#     ...  # train step
