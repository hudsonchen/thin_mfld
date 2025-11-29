import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def eval_covertype(sim, xT, data, loss):
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

    print("Final Train pred:", sim._vm_q1(data["Z"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    print("Final Train label:", data["y"][0, ...][:5].squeeze())
    print("Final Test pred:", sim._vm_q1(data["Z_test"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    print("Final Test label:", data["y_test"][0, ...][:5].squeeze())
    print("Final Train Loss:", train_losses[-1])
    print("Final Test Loss:", test_losses[-1])
    print("Final Train Acc:", train_accs[-1])
    print("Final Test Acc:", test_accs[-1])

    jnp.save(f'{args.save_path}/trajectory.npy', xT)
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/train_losses.npy', train_losses)
    jnp.save(f'{args.save_path}/test_losses.npy', test_losses)


def eval_vlm(args, sim, xT, data, init, x_ground_truth, 
             lotka_volterra_ws, lotka_volterra_ms, 
             mmd_path, thin_original_mse_path):
    rng_key = jax.random.PRNGKey(14)
    data_longer = lotka_volterra_ms(init, x_ground_truth, rng_key, end=100, noise_scale=0.)
    loss = jnp.zeros(xT.shape[0])
    for t, particles in enumerate(xT):
        # Run trajectories once
        rng_key, _ = jax.random.split(rng_key)
        sampled_trajectories_all = jax.vmap(lambda p: lotka_volterra_ws(init, p, rng_key, 100))(particles)
        sampled_trajectories = sampled_trajectories_all.mean(axis=0)
        sampled_trajectories_std = sampled_trajectories_all.std(axis=0)

        loss = loss.at[t].set(jnp.mean((sampled_trajectories - data_longer) ** 2, axis=(0,)).sum())
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

    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='MSE Loss')
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.savefig(f'{args.save_path}/vlm_loss.png')
    plt.close()

    jnp.save(f'{args.save_path}/vlm_loss.npy', loss)

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
    
    return