import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.kernel import *

def eval_boston(sim, xT, data, loss):
    train_losses = []
    test_losses = []
    for p in tqdm(xT):
        tr_, te_ = 0.0, 0.0
        for z_tr, y_tr in zip(data["Z"], data["y"]):
            tr_ += loss(z_tr, y_tr, p)
        for z_te, y_te in zip(data["Z_test"], data["y_test"]):
            te_ += loss(z_te, y_te, p)
        train_losses.append(float(tr_) / data["num_batches_tr"])
        test_losses.append(float(te_) / data["num_batches_te"])
    
    train_losses = jnp.array(train_losses)
    test_losses = jnp.array(test_losses)
    print("Final Train pred:", sim._vm_q1(data["Z"][0, ...], xT[-1]).mean(axis=0).squeeze()[:5])
    print("Final Train label:", data["y"][0, ...][:5].squeeze())
    print("Final Test pred:", sim._vm_q1(data["Z_test"][0, ...], xT[-1]).mean(axis=0).squeeze()[:5])
    print("Final Test label:", data["y_test"][0, ...][:5].squeeze())
    print("Final Train Loss:", train_losses[-1])
    print("Final Test Loss:", test_losses[-1])
    return train_losses, test_losses


def eval_covertype(args, sim, xT, data, loss, mmd_path, thin_original_mse_path,
                   time_path):
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    for p in tqdm(xT):
        tr_loss, tr_acc = 0.0, 0.0
        te_loss, te_acc = 0.0, 0.0
        for z_tr, y_tr in zip(data["Z"], data["y"]):
            l, a = loss(z_tr, y_tr, p)
            tr_loss += l
            tr_acc += a
        for z_te, y_te in zip(data["Z_test"], data["y_test"]):
            l, a = loss(z_te, y_te, p)
            te_loss += l
            te_acc += a
        train_losses.append(float(tr_loss) / data["num_batches_tr"])
        train_accs.append(float(tr_acc) / data["num_batches_tr"])
        test_losses.append(float(te_loss) / data["num_batches_te"])
        test_accs.append(float(te_acc) / data["num_batches_te"])
    train_losses = jnp.array(train_losses)
    train_accs = jnp.array(train_accs)
    test_losses = jnp.array(test_losses)
    test_accs = jnp.array(test_accs)

    # print("Final Train pred:", sim._vm_q1(data["Z"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    # print("Final Train label:", data["y"][0, ...][:5].squeeze())
    # print("Final Test pred:", sim._vm_q1(data["Z_test"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    # print("Final Test label:", data["y_test"][0, ...][:5].squeeze())
    # print("Final Train Loss:", train_losses[-1])
    # print("Final Test Loss:", test_losses[-1])
    # print("Final Train Acc:", train_accs[-1])
    # print("Final Test Acc:", test_accs[-1])

    jnp.save(f'{args.save_path}/trajectory.npy', xT)
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/train_losses.npy', train_losses)
    jnp.save(f'{args.save_path}/test_losses.npy', test_losses)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)

def eval_vlm(args, sim, xT, data, init, x_ground_truth, 
             lotka_volterra_ws, lotka_volterra_ms, 
             mmd_path, thin_original_mse_path, time_path):
    rng_key = jax.random.PRNGKey(14)
    data_longer = lotka_volterra_ms(init, x_ground_truth, rng_key, end=100, noise_scale=0.)
    loss = jnp.zeros(xT.shape[0])
    kgd_values = jnp.zeros(xT.shape[0])

    for t, particles in enumerate(tqdm(xT)):
        # Run trajectories once
        rng_key, _ = jax.random.split(rng_key)
        sampled_trajectories_all = jax.vmap(lambda p: lotka_volterra_ws(init, p, rng_key, 100))(particles)
        sampled_trajectories = sampled_trajectories_all.mean(axis=0)
        sampled_trajectories_std = sampled_trajectories_all.std(axis=0)

        loss = loss.at[t].set(jnp.mean((sampled_trajectories - data_longer) ** 2, axis=(0,)).sum())

        l = jax.jit(lambda x, y: k_imq(x, y, 1, 0.5,0.1)) #matern_kernel(x,y,5.0))
        alpha = 2.0
        beta = 1.0
        k = lambda x,y : recommended_kernel(x,y,l,alpha,beta,1.0)
        def S_PQ(X):
            N = X.shape[0]
            keys_outer = jax.random.split(rng_key, num=N)                # (N, 2)
            keys_mat = jax.vmap(lambda keys: jax.random.split(keys, num=N))(keys_outer)  # (N, N, 2)
            return -args.zeta * X + sim._vm_grad_q2(X, X, keys_mat).mean(axis=1)
        k_pq = GradientKernel(S_PQ, k)
        kgd = KernelGradientDiscrepancy(k_pq)
        kgd_value = kgd.evaluate(particles)
        kgd_values = kgd_values.at[t].set(kgd_value)

        if t % 10 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[1].plot(sampled_trajectories[:, 0], color='red', label='Prey')
            axes[1].plot(sampled_trajectories[:, 1], color='blue', label='Predator')
            axes[1].fill_between(
                jnp.arange(sampled_trajectories.shape[0]),
                sampled_trajectories[:, 0] - 2 * sampled_trajectories_std[:, 0],
                sampled_trajectories[:, 0] + 2 * sampled_trajectories_std[:, 0],
                color='red',
                alpha=0.3,
            )
            axes[1].fill_between(
                jnp.arange(sampled_trajectories.shape[0]),
                sampled_trajectories[:, 1] - 2 * sampled_trajectories_std[:, 1],
                sampled_trajectories[:, 1] + 2 * sampled_trajectories_std[:, 1],
                color='blue',
                alpha=0.3,
            )
            axes[1].plot(data_longer[:, 0], color='grey', linestyle='dashed', label='Ground Truth')
            axes[1].plot(data_longer[:, 1], color='grey', linestyle='dashed')
            axes[1].scatter(jnp.arange(data.shape[0]), data[:, 0], color='black', s=10, label='Prey Data')
            axes[1].scatter(jnp.arange(data.shape[0]), data[:, 1], color='black', s=10, label='Predator Data')
            axes[1].legend()
            axes[1].grid(True)
            plt.savefig(f'{args.save_path}/vlm_distribution_step_{t}.png')
            plt.close(fig)

    jnp.save(f'{args.save_path}/vlm_trajectory.npy', xT)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot
    axes[0].plot(loss, label='MSE Loss')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training Step')
    axes[0].legend()

    # Right subplot (example — duplicate or plot another metric)
    axes[1].plot(kgd_values, label='KGD')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Training Step')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{args.save_path}/vlm_loss.png')
    plt.close()


    jnp.save(f'{args.save_path}/vlm_loss.npy', loss)
    jnp.save(f'{args.save_path}/vlm_kgd.npy', kgd_values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- (2) MMD^2 path ---
    axes[0].plot(mmd_path, color="C2", label="MMD$^2$")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("MMD$^2$")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --- (3) Thinned vs Original MSE path ---
    axes[1].plot(thin_original_mse_path, color="C3", label="Thin-Original MSE")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Thinned vs Original Output MSE")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    # --- Layout and save ---
    plt.tight_layout()
    plt.savefig(f"{args.save_path}/mfld_debug_vector_field.png", dpi=300)
    plt.show()

    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    return


def eval_mmd_flow(args, sim, xT, data, mmd_path, thin_original_mse_path, time_path):
    @jax.jit
    def mmd_func(Y):
        K_XX = sim.problem.distribution.mean_mean_embedding()
        K_YY = sim.kernel.make_distance_matrix(Y, Y)
        K_XY = sim.problem.distribution.mean_embedding(Y)
        return jnp.sqrt(K_XX + K_YY.mean() - 2 * K_XY.mean())

    mmd_values = jnp.zeros(xT.shape[0])
    for t, particles in enumerate(tqdm(xT)):
        mmd_value = mmd_func(particles)
        mmd_values = mmd_values.at[t].set(mmd_value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(mmd_values)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('MMD Value')
    axes[1].plot(mmd_path)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('MMD between thinned set and original set')
    plt.savefig(f'{args.save_path}/mmd_values_trajectory.png')
    plt.close(fig)

    freq = 10
    jnp.save(f'{args.save_path}/mmd_flow_trajectory.npy', xT[::freq, :, :])
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/mmd_values_trajectory.npy', mmd_values)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    save_animation_2d(xT, sim.problem.distribution, save_path=args.save_path)
    return 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def save_animation_2d(
    trajectory,
    distribution,
    save_path,
    duration_sec=6.0,   # fixed total video length
    fps=20,             # fixed fps
    interval_ms=50,     # display interval (doesn't affect saved mp4 if fps is set)
    resolution=100,
    xlim=(-5, 5),
    ylim=(-5, 5),
):
    """
    Save an animation with fixed duration, regardless of trajectory length.

    trajectory: array of shape (T, N, 2)
    duration_sec: desired video length in seconds
    fps: frames per second for the saved mp4
    """
    T = int(trajectory.shape[0])
    if T <= 0:
        raise ValueError("trajectory must have at least one timestep")

    # Fixed number of frames in the output video
    num_frames = max(int(round(duration_sec * fps)), 1)

    # Map frame indices -> trajectory indices (monotone, covers [0, T-1])
    if num_frames == 1:
        t_idx = np.array([0], dtype=int)
    else:
        t_idx = np.linspace(0, T - 1, num_frames)
        t_idx = np.round(t_idx).astype(int)
        t_idx = np.clip(t_idx, 0, T - 1)

    # ---- create initial plot ----
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Background: logpdf on a grid
    x_vals = np.linspace(xlim[0], xlim[1], resolution)
    y_vals = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # If distribution.pdf expects jax arrays, swap np -> jnp here; otherwise keep numpy.
    logpdf = np.log(distribution.pdf(grid)).reshape(resolution, resolution)

    ax.imshow(logpdf, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")

    # NOTE: you had (y,x) swapped in scatter; keep consistent with your original intent.
    scat = ax.scatter(trajectory[0, :, 1], trajectory[0, :, 0], label="source")

    def update(frame_j):
        t = t_idx[frame_j]
        # set_offsets expects (N, 2) with columns [x, y]
        pts = np.asarray(trajectory[t])
        scat.set_offsets(np.column_stack([pts[:, 1], pts[:, 0]]))
        return (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=True,
        interval=interval_ms,
    )

    ani.save(f"{save_path}/animation.mp4", writer="ffmpeg", fps=fps)
    plt.close(fig)
    return


# def save_animation_2d(args, trajectory, kernel, distribution, rate, save_path):
#     T = trajectory.shape[0]
#     Y = trajectory[0, :, :]

#     num_timesteps = trajectory.shape[0]
#     num_frames = max(num_timesteps // rate, 1)

#     def update(frame):
#         _animate_scatter.set_offsets(trajectory[frame * rate, :, :])
#         return (_animate_scatter,)

#     # create initial plot
#     animate_fig, animate_ax = plt.subplots()
#     animate_ax.set_xlim(-5, 5)
#     animate_ax.set_ylim(-5, 5)
#     x_range = (-5, 5)
#     y_range = (-5, 5)
#     resolution = 100
#     x_vals = jnp.linspace(x_range[0], x_range[1], resolution)
#     y_vals = jnp.linspace(y_range[0], y_range[1], resolution)
#     X, Y = jnp.meshgrid(x_vals, y_vals)
#     grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
#     logpdf = jnp.log(distribution.pdf(grid).reshape(resolution, resolution))
#     plt.imshow(logpdf, extent=(-5, 5, -5, 5), origin='lower')

#     _animate_scatter = animate_ax.scatter(trajectory[0, :, 1], trajectory[0, :, 0], label='source')

#     ani = FuncAnimation(
#         animate_fig,
#         update,
#         frames=num_frames,
#         # init_func=init,
#         blit=True,
#         interval=50,
#     )
#     ani.save(f'{save_path}/animation.mp4',
#                    writer='ffmpeg', fps=20)
#     return    
