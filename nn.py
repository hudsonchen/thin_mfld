# !pip install jax jaxlib optax scikit-learn numpy tqdm

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from utils.datasets import load_boston

def glorot_normal(key, fan_in, fan_out):
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return std * jax.random.normal(key, (fan_in, fan_out))

def init_params(key, d_in, d_hidden, d_out=1):
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "W1": glorot_normal(k1, d_in, d_hidden),
        "b1": jnp.zeros((d_hidden,)),
        "W2": glorot_normal(k2, d_hidden, d_out),
    }
    return params

def mlp_apply(params, x):
    """x: (batch, d_in) -> (batch,)"""
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    y = h @ params["W2"]
    return y.squeeze(-1) / jnp.sqrt(h.shape[-1])

# -----------------------------
# Loss & metrics
# -----------------------------
def mse_loss(params, X, y):
    preds = mlp_apply(params, X)
    return jnp.mean((preds - y) ** 2)


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)

# -----------------------------
# Training loop
# -----------------------------


def train(
    seed=0,
    hidden=64,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=64,
    epochs=200,
    standardize_X=True,
    standardize_y=False,
):
    # Load data
    
    data = load_boston(batch_size, standardize_X=True, standardize_y=False)
    X_tr, y_tr = data["Z"], data["y"]
    X_te, y_te = data["Z_test"], data["y_test"]

    key = jax.random.PRNGKey(seed)
    d_in = X_tr.shape[-1]

    # Init model & optimizer
    params = init_params(key, d_in, hidden)
    optimizer = optax.sgd(learning_rate=lr * params["W1"].shape[0])
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, X, y):
        loss_value, grads = jax.value_and_grad(mse_loss)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
        
    # Train
    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        key, sk = jax.random.split(key)
        # One pass over shuffled minibatches
        batch_losses = []

        for Xb, yb in zip(X_tr, y_tr):   # iterate over pre-batched data
            params, opt_state, loss_value = update(params, opt_state, Xb, yb)
            batch_losses.append(loss_value)

        train_loss = float(jnp.mean(jnp.array(batch_losses)))

        # Eval each epoch
        with jax.disable_jit():
            preds_te = mlp_apply(params, X_te)
            mse_val = float(mse(preds_te, y_te))

        pbar.set_postfix(mse_train=train_loss, mse_val=mse_val)

    # Final evaluation (return params too)
    return {
        "params": params,
        "mse": mse_val,
    }

if __name__ == "__main__":
    results = train(
        seed=0,
        hidden=1024,         # try 32/64/128
        lr=1e-3,
        weight_decay=1e-5, # small L2 helps
        batch_size=64,
        epochs=300,
        standardize_X=True,
        standardize_y=False,  # often fine to keep target unscaled for reporting
    )
