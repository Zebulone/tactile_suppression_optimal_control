import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from suppression_oc.constants import (
    INIT_SIGMA_POS_NORMAL,
    INIT_SIGMA_POS_UNCERTAIN,
    CONDITIONS,
)

from suppression_oc.oc.control import glqg
from suppression_oc.oc.models.todorov import make_delayed_reaching_task_plus

detection = pd.read_csv("./data/tatai_et_al_detection.csv")


Ks = []
ts = []

for subject in detection["subject"].unique():
    reach_time = detection.loc[detection["subject"] == subject, "reach_time"].mean()
    reach_time = int(reach_time * 100)

    sigma0s = [INIT_SIGMA_POS_NORMAL, INIT_SIGMA_POS_UNCERTAIN]

    for sigma0, condition in zip(sigma0s, CONDITIONS):
        Sigma0 = jnp.eye(1 * 4) * jnp.array(1 * [sigma0, 1e-5, 1e-5, 1e-5])
        x0 = jnp.array(1 * [0.5, 0, 0, 0])
        todorov_spec = make_delayed_reaching_task_plus(
            T=21 + reach_time,
            initial_period=20,
            final_period=1,
        )
        _, K, L, _, _ = glqg.solve(
            lqgspec=todorov_spec,
            x0=x0,
            Sigma0=Sigma0 @ Sigma0.T,
        )

        key = jax.random.PRNGKey(0)
        trajectories, inov = glqg.simulate_different_start(
            key, 100, todorov_spec, K, L, x0=x0, start_offset=0.03, delay=0
        )

        K_frame = {}
        K_frame["time"] = jnp.arange(K.shape[0]) * 0.01
        K_frame["percent_time"] = np.append(
            np.arange(-0.2, 0.0, 0.01), np.linspace(0, 1, K.shape[0] - 20)
        )
        K_frame["K_pos"] = K[:, 0, 0]
        K_frame["K_vel"] = K[:, 1, 0]
        K_frame["K_acc"] = K[:, 2, 0]
        K_frame["subject"] = subject
        K_frame["condition"] = condition

        Ks.append(pd.DataFrame(K_frame))

        # store all trajectories efficiently: shape (n_sims*T) rows with reach_i identifier
        n_sims, T, _ = trajectories.shape
        time = np.arange(T) * 0.01
        percent_time = np.append(np.arange(-0.2, 0.0, 0.01), np.linspace(0, 1, T - 20))

        df_traj = pd.DataFrame(
            {
                "reach_i": np.repeat(np.arange(n_sims, dtype=np.int32), T),
                "time": np.tile(time, n_sims).astype(np.float32),
                "percent_reach_t": np.tile(percent_time, n_sims).astype(np.float32),
                "d": trajectories[:, :, 0].reshape(-1).astype(np.float32),
                "v": trajectories[:, :, 1].reshape(-1).astype(np.float32),
                "a": trajectories[:, :, 2].reshape(-1).astype(np.float32),
                "subject": np.repeat(np.array([subject]), n_sims * T),
                "condition": np.repeat(np.array([condition]), n_sims * T),
            }
        )
        ts.append(df_traj)


K_df = pd.concat(Ks, ignore_index=True)
K_df.to_csv("./data/kalman_gains_initial_uncertainty.csv", index=False)

T_df = pd.concat(ts, ignore_index=True)
T_df["v"] *= -1
T_df["a"] *= -1
T_df.to_csv("./data/reaches_simulated.csv", index=False)
