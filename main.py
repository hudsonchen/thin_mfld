from utils.configs import CFG
from utils.problems import Problem
from mfld import MFLD
from utils.datasets import load_boston
import jax.numpy as jnp
import jax
from tqdm import tqdm

def R1_prime(s):  # R1(s)=0.5*s^2
    return s

def q1_nn(z, x):
    # Simple 2-layer NN for demonstration
    d_hidden = x.shape[0] - 3
    W1, b1, W2, b2 = x[:d_hidden], x[d_hidden+1], x[d_hidden+2], x[d_hidden+3]
    # assert W1.shape[0] == z.shape[0]
    h = jnp.tanh(z @ W1 + b1)
    return jnp.dot(W2, h) + b2


data = load_boston(standardize_X=True, standardize_y=False)


problem_nn = Problem(
    d = data["Z"].shape[1] + 3, 
    R1_prime=R1_prime,
    q1=q1_nn,
    q2=None,
    gradx_q2=None,
    data=data
)


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)  # optional for stability

    cfg = CFG(N=64, steps=5000, step_size=1e-3, sigma=0.001, seed=0, return_path=True)
    sim = MFLD(cfg, problem_nn)
    xT = sim.simulate()

    def compute_mse(X, y, params):
        """Compute MSE for a given parameter vector `params`."""
        preds = jax.vmap(q1_nn, in_axes=(None, 0))(X, params).mean(axis=0)
        return jnp.mean((preds - y) ** 2)

    # Plotting code
    T_plus_1, N, d = xT.shape

    # vmap over particles
    loss_fn = lambda p: (
        compute_mse(data["Z"], data["y"], p),
        compute_mse(data["Z_test"], data["y_test"], p)
    )

    xT_subsampled = xT[::50]  # Subsample for plotting
    train_losses, test_losses = jax.vmap(loss_fn)(xT_subsampled)

    train_losses = jnp.array(train_losses)
    test_losses = jnp.array(test_losses)

    # ---- Plot ----
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses, label="Test MSE")
    plt.xlabel("Training step")
    plt.ylabel("Mean Squared Error")
    plt.title("Training / Test Loss vs Step")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("results/mfld_boston_nn_loss.png")
