from utils.configs import CFG
from utils.problems import Problem
from mfld import MFLD
from utils.datasets import load_boston
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import lax


def R1_prime(s):  # R1(s)=0.5*s^2
    return s

def q1_nn(z, x):
    # Simple 2-layer NN for demonstration
    d_hidden = x.shape[0] - 3
    W1, b1, W2, b2 = x[:d_hidden], x[d_hidden+1], x[d_hidden+2], x[d_hidden+3]
    h = jnp.tanh(z @ W1 + b1)
    return jnp.dot(W2, h) + b2


data = load_boston(batch_size=64, standardize_X=True, standardize_y=False)


problem_nn = Problem(
    particle_d=data["Z"].shape[-1] + 3,  # NN params dimension
    data_d=data["Z"].shape[-1],
    R1_prime=R1_prime,
    q1=q1_nn,
    q2=None,
    gradx_q2=None,
    data=data
)


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)  # optional for stability

    cfg = CFG(N=128, steps=50000, step_size=1.0, sigma=1.0, zeta=1e-2, seed=0, return_path=True)
    sim = MFLD(cfg, problem_nn)
    xT = sim.simulate()

    def compute_mse(Z, y, params):
        """Compute MSE for a given parameter vector `params`."""
        preds_all = jax.vmap(q1_nn, in_axes=(None, 0))(Z, params)
        preds = preds_all.mean(axis=0)
        return jnp.mean((preds - y) ** 2)

    # Plotting code
    T_plus_1, N, d = xT.shape

    # vmap over particles
    loss_fn = lambda p: (
        compute_mse(data["Z"][0, ...], data["y"][0, ...], p),
        compute_mse(data["Z_test"], data["y_test"], p)
    )

    xT_subsampled = xT[::100]  # Subsample for plotting
    # train_losses, test_losses = jax.vmap(loss_fn)(xT_subsampled)
    outs = lax.map(lambda p: jnp.array(loss_fn(p)), xT_subsampled)  # (N, 2)
    train_losses, test_losses = outs[:, 0], outs[:, 1]

    train_losses = jnp.array(train_losses)
    test_losses = jnp.array(test_losses)

    print("Final Train pred:", jax.vmap(q1_nn, in_axes=(None, 0))(data["Z"][0, ...], xT_subsampled[-1]).mean(axis=0)[:10])
    print("Final Train label:", data["y"][0, ...][:10])
    print("Final Test pred:", jax.vmap(q1_nn, in_axes=(None, 0))(data["Z_test"], xT_subsampled[-1]).mean(axis=0)[:10])
    print("Final Test label:", data["y_test"][:10])
    print("Final Train MSE:", train_losses[-1])
    print("Final Test MSE:", test_losses[-1])

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
