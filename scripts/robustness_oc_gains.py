from jax import jit
import jax.numpy as jnp
import numpy as np

np.random.seed(42)
import pandas as pd

from suppression_oc.oc.control import glqg
from suppression_oc.oc.models.todorov import (
    make_delayed_reaching_task_plus,
    DEFAULT_SIGMA_POS,
    DEFAULT_SIGMA_VEL,
    DEFAULT_SIGMA_FRC,
    DEFAULT_C,
    DEFAULT_R,
    DEFAULT_V,
    DEFAULT_F,
)
from suppression_oc.constants import INIT_SIGMA_POS_NORMAL

LOWER_BOUND = 0.1
UPPER_BOUND = 10.0

# Increase if needed, 1000 combinations were used in the paper
N_COMBINATIONS = 1000


def solve(sigma_pos, sigma_vel, sigma_frc, c, r, v, f, sigma0):
    """Solve the optimal control problem for the given parameters."""
    Sigma0 = jnp.eye(4) * jnp.array([sigma0, 1e-5, 1e-5, 1e-5])
    x0 = jnp.array([0.5, 0.0, 0.0, 0.0])

    task = make_delayed_reaching_task_plus(
        sigma_pos=sigma_pos,
        sigma_vel=sigma_vel,
        sigma_frc=sigma_frc,
        c=c,
        r=r,
        v=v,
        f=f,
    )

    _, K, _, _, _ = glqg.solve(
        lqgspec=task,
        x0=x0,
        Sigma0=Sigma0 @ Sigma0.T,
    )
    return K[:, 0, 0]


for i in range(1, N_COMBINATIONS + 1):
    run = {}
    run["i"] = i
    run["sigma_pos"] = DEFAULT_SIGMA_POS * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["sigma_vel"] = DEFAULT_SIGMA_VEL * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["sigma_frc"] = DEFAULT_SIGMA_FRC * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["c"] = DEFAULT_C * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["r"] = DEFAULT_R * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["v"] = DEFAULT_V * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["f"] = DEFAULT_F * np.random.uniform(LOWER_BOUND, UPPER_BOUND)
    run["sigma0"] = INIT_SIGMA_POS_NORMAL * np.random.uniform(LOWER_BOUND, UPPER_BOUND)

    Sigma0 = jnp.eye(4) * jnp.array([run["sigma0"], 1e-5, 1e-5, 1e-5])
    x0 = jnp.array([0.5, 0.0, 0.0, 0.0])

    K = solve(
        sigma_pos=run["sigma_pos"],
        sigma_vel=run["sigma_vel"],
        sigma_frc=run["sigma_frc"],
        c=run["c"],
        r=run["r"],
        v=run["v"],
        f=run["f"],
        sigma0=run["sigma0"],
    )
    for K_i, K_val in enumerate(K):
        run[f"K_pos_{K_i}"] = float(K_val)

    df = pd.DataFrame([run])
    df.to_csv(
        "./data/robustness_oc_gains.csv",
        mode="a",
        header=not pd.io.common.file_exists("./data/robustness_oc_gains.csv"),
        index=False,
    )
