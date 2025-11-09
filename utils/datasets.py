# pip install scikit-learn
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax
import jax.numpy as jnp

def load_boston(batch_size, test_size=0.2, seed=42, standardize_X=True, standardize_y=False):
    """
    Returns JAX arrays and (optionally) standardization stats.
    Shapes:
      X_*: (N, 13), y_*: (N, 1)  # regression target
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

    y_min = y_tr.min(axis=0, keepdims=True)
    y_max = y_tr.max(axis=0, keepdims=True)
    if standardize_y:
        y_tr = 2 * (y_tr - y_min) / (y_max - y_min + 1e-8) - 1
        y_te = 2 * (y_te - y_min) / (y_max - y_min + 1e-8) - 1

    # Shuffle the training data before batching
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, X_tr.shape[0])
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]

    # Batch them
    N = y_tr.shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    pad = num_batches * batch_size - N
    if pad > 0:
        y_tr = jnp.pad(y_tr, ((0, pad), (0,0)), mode='edge')
        X_tr = jnp.pad(X_tr, ((0, pad), (0,0)), mode='edge')
    y_tr = y_tr.reshape(num_batches, batch_size, *y_tr.shape[1:])
    X_tr = X_tr.reshape(num_batches, batch_size, *X_tr.shape[1:])

    # Convert to JAX arrays (targets flattened to shape (N,))
    out = {
        "Z": jnp.asarray(X_tr),
        "y": jnp.asarray(y_tr.squeeze(-1)),
        "Z_test":  jnp.asarray(X_te),
        "y_test":  jnp.asarray(y_te.squeeze(-1)),
        "z_stats": (jnp.asarray(x_mean.squeeze(0)), jnp.asarray(x_std.squeeze(0))),
        "y_stats": (jnp.asarray(y_min.item()),     jnp.asarray(y_max.item())),
        "batch_size": batch_size
    }
    return out

def load_covertype(batch_size, test_size=0.2, seed=42,
                   standardize_X=True, one_hot_y=False):
    """
    Returns batched JAX arrays for UCI Covertype classification.
    Shapes:
      X: (num_batches, batch_size, 54)
      y: (num_batches, batch_size,) or one-hot
      X_test: (N_test, 54)
      y_test: same
    """
    
    # ---- Load dataset ----
    ds = fetch_covtype(as_frame=False)
    X = ds.data.astype(np.float32)
    y = (ds.target - 1).astype(np.int32)  # convert to {0,...,6}

    # ---- Train/test split ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ---- Standardization ----
    if standardize_X:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
        x_mean = scaler.mean_.astype(np.float32)
        x_std  = scaler.scale_.astype(np.float32)
    else:
        x_mean = X_tr.mean(axis=0).astype(np.float32)
        x_std  = X_tr.std(axis=0).astype(np.float32) + 1e-8

    # ---- One-hot encoding (optional) ----
    if one_hot_y:
        num_classes = 7
        y_tr_oh = np.eye(num_classes)[y_tr]
        y_te_oh = np.eye(num_classes)[y_te]

        y_tr = y_tr_oh.astype(np.float32)
        y_te = y_te_oh.astype(np.float32)

    # ---- Shuffle training set ----
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, X_tr.shape[0])
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]

    # ---- Batch + pad ----
    N_train = X_tr.shape[0]
    N_test = X_te.shape[0]
    num_batches_tr = (N_train + batch_size - 1) // batch_size
    num_batches_te = (N_test + batch_size - 1) // batch_size
    pad_tr = num_batches_tr * batch_size - N_train
    pad_te = num_batches_te * batch_size - N_test
    if pad_tr > 0:
        X_tr = jnp.pad(X_tr, ((0, pad_tr), (0, 0)), mode='edge')
        if one_hot_y:
            y_tr = jnp.pad(y_tr, ((0, pad_tr), (0, 0)), mode='edge')
        else:
            y_tr = jnp.pad(y_tr, ((0, pad_tr),), mode='edge')

    X_tr = X_tr.reshape(num_batches_tr, batch_size, -1)
    if one_hot_y:
        y_tr = y_tr.reshape(num_batches_tr, batch_size, -1)
    else:
        y_tr = y_tr.reshape(num_batches_tr, batch_size)

    if pad_te > 0:
        X_te = jnp.pad(X_te, ((0, pad_te), (0, 0)), mode='edge')
        if one_hot_y:
            y_te = jnp.pad(y_te, ((0, pad_te), (0, 0)), mode='edge')
        else:
            y_te = jnp.pad(y_te, ((0, pad_te),), mode='edge')
    X_te = X_te.reshape(num_batches_te, batch_size, -1)
    if one_hot_y:
        y_te = y_te.reshape(num_batches_te, batch_size, -1)
    else:
        y_te = y_te.reshape(num_batches_te, batch_size)

    # ---- Convert to JAX arrays ----
    out = {
        "Z": jnp.asarray(X_tr),
        "y": jnp.asarray(y_tr),
        "Z_test": jnp.asarray(X_te),
        "y_test": jnp.asarray(y_te),
        "z_stats": (jnp.asarray(x_mean), jnp.asarray(x_std)),
        "num_classes": 7,
        "batch_size": batch_size,
        "num_batches_tr": num_batches_tr,
        "num_batches_te": num_batches_te,
    }

    return out

