import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp

from suppression_oc.constants import (
    INIT_SIGMA_POS_NORMAL,
    CM_TO_INCHES,
    CONDITIONS,
    CONDITION_COLORS,
)
from suppression_oc.util import ci95

from suppression_oc.oc.models.todorov import (
    DEFAULT_SIGMA_POS,
    DEFAULT_SIGMA_VEL,
    DEFAULT_SIGMA_FRC,
    DEFAULT_C,
    DEFAULT_R,
    DEFAULT_V,
    DEFAULT_F,
    make_delayed_reaching_task_plus,
)

from suppression_oc.oc.control import glqg


def plot_mean_std(
    data,
    t,
    var,
    ax,
    dt=0.01,
    plot_std=False,
    color=None,
    label=None,
    linestyle="-",
):
    """Plot the mean and standard deviation of a variable binned over time."""

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = np.arange(dt / 2, 1.01 - dt / 2, dt)
    bins = np.arange(0, 1.01, dt)

    ax.plot(
        x,
        data.groupby(pd.cut(data[t], bins=bins), observed=True)[var].mean(),
        label=label,
        color=color,
        linestyle=linestyle,
    )

    if plot_std:
        ax.fill_between(
            x,
            data.groupby(pd.cut(data[t], bins=bins), observed=True)[var].mean()
            - data.groupby(pd.cut(data[t], bins=bins), observed=True)[var].std(),
            data.groupby(pd.cut(data[t], bins=bins), observed=True)[var].mean()
            + data.groupby(pd.cut(data[t], bins=bins), observed=True)[var].std(),
            color=color,
            alpha=0.2,
        )


def plot_subject_mean_condition(data, var):
    """Plot the mean of a variable for each subject and condition."""

    fig, ax = plt.subplots(1, 1, figsize=(1.3 * CM_TO_INCHES, 2 * CM_TO_INCHES))

    reaction_time_mean = (
        data.groupby(["condition", "subject"])[var].mean()
    ).reset_index(name="mean")

    for subject in reaction_time_mean.subject.unique():
        ax.plot(
            [0, 1],
            reaction_time_mean[reaction_time_mean.subject == subject]["mean"],
            color="gray",
            alpha=0.2,
            marker="o",
        )

    for condition, idx in zip(CONDITIONS[:2], [0.1, 1.1]):
        condition_means = reaction_time_mean[
            reaction_time_mean.condition == condition
        ]["mean"]
        ax.errorbar(
            [idx],
            [condition_means.mean()],
            yerr=ci95(len(condition_means)) * condition_means.sem(),
            color=CONDITION_COLORS[condition],
            fmt="o",
            zorder=3,
        )

    ax.set(
        xticks=[0, 1],
        xticklabels=["Norm.", "Uncert."],
    )

    ax.tick_params(axis="x", rotation=20)

    return fig, ax


def solve_for_param_scaled(param, scales):

    base_values = {
        "sigma_pos": float(DEFAULT_SIGMA_POS),
        "sigma_vel": float(DEFAULT_SIGMA_VEL),
        "sigma_frc": float(DEFAULT_SIGMA_FRC),
        "v": DEFAULT_V,
        "f": DEFAULT_F,
        "r": DEFAULT_R,
        "c": DEFAULT_C,
    }

    base_sigma0 = INIT_SIGMA_POS_NORMAL

    K_list = []
    labels = []
    for s in scales:
        kwargs = base_values.copy()

        if param == "sigma0":
            Sigma0 = jnp.eye(4) * jnp.array([s * base_sigma0, 1e-5, 1e-5, 1e-5])
        else:
            Sigma0 = jnp.eye(4) * jnp.array([base_sigma0, 1e-5, 1e-5, 1e-5])
            kwargs[param] = base_values[param] * s

        spec = make_delayed_reaching_task_plus(**kwargs)
        x0 = jnp.array([0.5, 0.0, 0.0, 0.0])
        _, K, _, _, _ = glqg.solve(lqgspec=spec, x0=x0, Sigma0=Sigma0 @ Sigma0.T)
        K_list.append(K)
        labels.append(f"x{s:g}")
    return K_list, labels
