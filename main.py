from utils.configs import CFG
from utils.problems import Problem_nn, Problem_vlm
from mfld import MFLD_nn, MFLD_vlm
from utils.datasets import load_boston, load_covertype
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import lax
import time
import os
import argparse
import pickle
from utils.lotka_volterra import lotka_volterra_ws, lotka_volterra_ms
from utils.evaluate import eval_boston, eval_covertype, eval_vlm

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use only 50% of GPU memory
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# from jax import config
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def get_config():
    parser = argparse.ArgumentParser(description='mmd_flow_cubature')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=str, default='sobolev')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--g', type=int, default=0)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=100)
    parser.add_argument('--particle_num', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--thinning', type=str, default='kt')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"neural_network_{args.dataset}/thinning_{args.thinning}/"
    args.save_path += f"kernel_{args.kernel}__step_size_{args.step_size}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__g_{args.g}__particle_num_{args.particle_num}__noise_scale_{args.noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args

def main(args):
    if args.dataset == 'boston':
        def R1_prime(hat_y, y):  # R1(s)=0.5*s^2
            return hat_y - y

        def q1_nn(z, x):
            d_hidden = z.shape[-1]
            W1, b1, W2 = x[:d_hidden], x[d_hidden+1], x[d_hidden+1:]
            h = jnp.tanh(z @ W1 + b1)
            return jnp.dot(W2, h)

        data = load_boston(batch_size=64, standardize_X=True, standardize_y=False)

        @jax.jit
        def loss(Z, y, params):
            """Compute MSE for a given parameter vector `params`."""
            preds_all = jax.vmap(                       # over particles
                    jax.vmap(q1_nn, in_axes=(0, None)),     # over batch
                    in_axes=(None, 0)                          # Z[p], params[p]
                )(Z, params)
            preds = preds_all.mean(axis=0)
            return jnp.mean((preds - y) ** 2)
        
        output_d = data["y"].shape[-1] if len(data["y"].shape) > 2 else 1
        input_d = data["Z"].shape[-1]
        problem_nn = Problem_nn(
            particle_d=data["Z"].shape[-1] + 1 + output_d,  # NN params dimension
            input_d=input_d,
            output_d=output_d,
            R1_prime=R1_prime,
            q1=q1_nn,
            q2=None,
            gradx_q2=None,
            data=data
        )

    elif args.dataset == 'covertype':

        def R1_prime(hat_y, y):  # R1(s)=0.5*s^2
            return - y / (hat_y + 1e-8)

        def q1_nn(z, x):
            d_hidden = z.shape[-1]
            W1, b1, W2 = x[:d_hidden], x[d_hidden+1], x[d_hidden+1:]
            h = jnp.tanh(z @ W1 + b1) 
            logits = jnp.dot(W2, h)
            return jax.nn.softmax(logits)

        data = load_covertype(batch_size=256, standardize_X=True, one_hot_y=True)

        @jax.jit
        def loss(Z, y, params):
            """Compute Cross-Entropy Loss for a given parameter vector `params`."""
            preds_all = jax.vmap(                       # over particles
                    jax.vmap(q1_nn, in_axes=(0, None)),     # over batch
                    in_axes=(None, 0)                          # Z[p], params[p]
                )(Z, params)
            preds = preds_all.mean(axis=0)  # (batch_size, num_classes)
            loss_val = -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-8), axis=1))
            acc_val = jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(y, axis=1))
            return loss_val, acc_val

        output_d = data["y"].shape[-1] if len(data["y"].shape) > 2 else 1
        input_d = data["Z"].shape[-1]
        problem_nn = Problem_nn(
            particle_d=data["Z"].shape[-1] + 1 + output_d,  # NN params dimension
            input_d=input_d,
            output_d=output_d,
            R1_prime=R1_prime,
            q1=q1_nn,
            q2=None,
            gradx_q2=None,
            data=data
        )

    elif args.dataset == 'vlm':
        from utils.kernel import gaussian_kernel
        # init = jnp.array([10.0, 15.0])
        init = jnp.array([10.0, 10.0])
        # x_ground_truth = jnp.array([-1., -1.5413]) # True parameters for Lotka-Volterra copied from Clementine's code
        x_ground_truth = jnp.array([-2.0, -1.733]) # True parameters from Zheyang's paper
        data = lotka_volterra_ws(init, x_ground_truth)
        def q2(x, x_prime):
            traj_1 = lotka_volterra_ms(init, x)
            traj_2 = lotka_volterra_ms(init, x_prime)
            kernel_fn = jax.vmap(jax.vmap(gaussian_kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))
            part1 = kernel_fn(traj_1, traj_2, 1.0)
            part2 = kernel_fn(traj_1, data, 1.0)
            return part1.sum() - 2 * part2.sum()
        
        problem_vlm = Problem_vlm(
            particle_d=2,
            q2=q2,
            data=data
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.dataset in ['boston', 'covertype']:
        cfg = CFG(N=args.particle_num, steps=args.step_num, step_size=args.step_size, sigma=args.noise_scale, kernel=args.kernel,
              zeta=1e-4, g=args.g, seed=args.seed, bandwidth=args.bandwidth, return_path=True)
        sim = MFLD_nn(problem=problem_nn, save_freq=data["num_batches_tr"], thinning=args.thinning, cfg=cfg)
        X0 = None
    elif args.dataset == 'vlm':
        cfg = CFG(N=args.particle_num, steps=args.step_num, step_size=args.step_size, sigma=args.noise_scale, kernel=args.kernel,
              zeta=1e-4, g=args.g, seed=args.seed, bandwidth=args.bandwidth, return_path=True)
        sim = MFLD_vlm(problem=problem_vlm, save_freq=1, thinning=args.thinning, cfg=cfg)
        rng_key = jax.random.PRNGKey(args.seed)
        X0 = jnp.stack([x_ground_truth] * args.particle_num, 0)
        X0 += 1e-5 * jax.random.normal(rng_key, X0.shape)
    xT, mmd_path, thin_original_mse_path = sim.simulate(x0=X0)

    # Plotting code
    T_plus_1, N, d = xT.shape

    if args.dataset == 'boston':
        eval_boston(sim, xT, data, loss)

    elif args.dataset == 'covertype':
        eval_covertype(sim, xT, data, loss)

    elif args.dataset == 'vlm':
        eval_vlm(args, sim, xT, data, init, x_ground_truth, lotka_volterra_ws, lotka_volterra_ms)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # # ---- Plot ----
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # # --- (1) Training and test losses ---
    # axes[0].plot(train_losses, label="Train Loss")
    # axes[0].plot(test_losses, label="Test Loss")
    # axes[0].set_ylabel("Loss")
    # axes[0].set_title("Training / Test Loss vs Step")
    # axes[0].legend()
    # axes[0].grid(True, linestyle="--", alpha=0.5)

    # # --- (2) MMD^2 path ---
    # axes[1].plot(mmd_path, color="C2", label="MMD$^2$")
    # axes[1].set_xlabel("Training Step")
    # axes[1].set_ylabel("MMD$^2$")
    # axes[1].legend()
    # axes[1].set_yscale("log")
    # axes[1].grid(True, linestyle="--", alpha=0.5)

    # # --- (3) Thinned vs Original MSE path ---
    # axes[2].plot(thin_original_mse_path, color="C3", label="Thin-Original MSE")
    # axes[2].set_xlabel("Training Step")
    # axes[2].set_ylabel("MSE")
    # axes[2].set_title("Thinned vs Original Output MSE")
    # axes[2].legend()
    # axes[2].set_yscale("log")
    # axes[2].grid(True, linestyle="--", alpha=0.5)
    # # --- Layout and save ---
    # plt.tight_layout()
    # plt.savefig(f"{args.save_path}/mfld_boston_nn_loss_mmd.png", dpi=300)
    # plt.show()

    return


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # optional for stability
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
