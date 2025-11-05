from utils.configs import CFG
from utils.problems import Problem
from mfld import MFLD
from utils.datasets import load_boston
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import lax
import time
import os
import argparse
import pickle

def get_config():
    parser = argparse.ArgumentParser(description='mmd_flow_cubature')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=100)
    parser.add_argument('--particle_num', type=int, default=100)
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"neural_network/uci/matern_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args



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
    jax.config.update("jax_enable_x64", True)  # optional for stability
    args = get_config()
    cfg = CFG(N=256, steps=5000, step_size=args.step_size, sigma=1.0, zeta=1e-3, seed=0, return_path=True)
    sim = MFLD(thinning=True, cfg=cfg, problem=problem_nn)
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

    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')
    new_save_path = args.save_path + '__complete'
    import shutil
    if os.path.exists(new_save_path):
        shutil.rmtree(new_save_path)  # Deletes existing folder
    os.rename(args.save_path, new_save_path)
    print(f'Job completed. Renamed folder to: {new_save_path}')
