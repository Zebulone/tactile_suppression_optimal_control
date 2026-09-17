import pandas as pd
from scipy.stats import norm, t
from scipy.signal import fftconvolve
from numpy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt

from suppression_oc.constants import (
    CONDITION_COLORS,
    CONDITION_LABELS_DICT,
    CONDITION_LINESTYLES,
    CM_TO_INCHES,
    TIMEPOINT_ORDER,
)


def ci95(n: int) -> float:
    """Multiplier for the 95% confidence interval of a mean over n observations

    Uses the t quantile (2.042 for n = 31), not the normal quantile 1.96.
    """
    return float(t.ppf(0.975, n - 1))


def signal_detection_parameters(data: pd.DataFrame) -> np.array:
    """caculate d' value for the given time bin"""
    false_alarm_rate = data["false_alarm"].sum() / data["intensity"].eq(0).sum()

    # add 0.5/N to avoid 0 false alarm rate
    if false_alarm_rate == 0:
        false_alarm_rate += 0.5 / data["intensity"].eq(0).sum()
    false_alarm_rate = norm.ppf(false_alarm_rate)

    bins = data.groupby(["timepoint_bin"], observed=True)
    hit_rates = bins.apply(
        lambda x: x["hit"].sum() / (x["intensity"] > 0).sum(), include_groups=False
    )
    n_stimuli = bins.apply(lambda x: (x["intensity"] > 0).sum(), include_groups=False)
    hit_rates.loc[hit_rates == 1] -= 0.5 / n_stimuli
    hit_rates.loc[hit_rates == 0] += 0.5 / n_stimuli
    hit_rates = hit_rates.apply(norm.ppf)

    d_primes = hit_rates - false_alarm_rate
    cs = -0.5 * (hit_rates + false_alarm_rate)
    return d_primes, cs


def norm_time_profiles(reaches_cond, norm_grid):
    """Average per-trial speed/acceleration profiles over normalized movement time"""
    signals = {"vel": ("v", False), "acc": ("a", True)}
    prof = {k: {} for k in signals}
    for subject, subj_df in reaches_cond.groupby("subject"):
        curves = {k: [] for k in signals}
        for _, trial in subj_df.groupby(["trial"]):
            trial = trial.sort_values("percent_reach_t")
            x = trial["percent_reach_t"].to_numpy()
            for k, (col, take_abs) in signals.items():
                y = trial[col].to_numpy()
                curves[k].append(
                    np.interp(
                        norm_grid,
                        x,
                        np.abs(y) if take_abs else y,
                        left=np.nan,
                        right=np.nan,
                    )
                )
        for k in signals:
            prof[k][subject] = np.nanmean(curves[k], axis=0)
    return {k: pd.DataFrame(prof[k], index=norm_grid).T for k in signals}


def continuous_profiles(profs_norm, mt, t_grid, dt, padding, subjects):
    """Stretch normalized profiles to each subject's movement time on a seconds grid"""
    n_pad = int(round(padding / dt))
    out = {k: {} for k in profs_norm}
    for s in subjects:
        n_move = int(round(mt[s] / dt))
        t_move = dt * np.arange(n_move + 1)
        frac = t_move / t_move[-1]
        t = np.concatenate(
            [
                dt * np.arange(-n_pad, 0),
                t_move,
                t_move[-1] + dt * np.arange(1, n_pad + 1),
            ]
        )
        idx = np.round((t - t_grid[0]) / dt).astype(int)
        keep = (idx >= 0) & (idx < len(t_grid))
        for k, prof in profs_norm.items():
            vals = np.interp(
                frac, prof.columns.values, prof.loc[s].values, left=np.nan, right=np.nan
            )
            padded = np.concatenate([np.zeros(n_pad), vals, np.zeros(n_pad)])
            col = np.full(len(t_grid), np.nan)
            col[idx[keep]] = padded[keep]
            out[k][s] = col
    return {k: pd.DataFrame(v, index=t_grid) for k, v in out.items()}


def prep_condition(data, condition):
    """Sort one condition's detection data into subject x timepoint-bin row order"""
    bin_order_map = {b: i for i, b in enumerate(TIMEPOINT_ORDER)}
    df = (
        data[data["condition"] == condition]
        .copy()
        .assign(bin_order=lambda x: x["timepoint_bin"].map(bin_order_map))
        .sort_values(["subject", "bin_order"])
        .reset_index(drop=True)
    )
    grp = df.groupby("timepoint_bin")["d_prime"]
    return (
        df,
        df["d_prime"].values,
        grp.mean().reindex(TIMEPOINT_ORDER).values,
        grp.sem().reindex(TIMEPOINT_ORDER).values,
    )


def bin_window(b, mts, move_bins, padding):
    """Absolute start/end time in seconds of a stimulation bin for one subject"""
    if b in move_bins:
        lo, hi = move_bins[b]
        return lo * mts, hi * mts
    return {
        "(-inf, -0.1]": (-padding, -0.1),
        "(-0.1, 0.0]": (-0.1, 0.0),
        "(1.0, inf]": (mts, mts + padding),
    }[b]


def bin_feature(
    profile_df,
    mt,
    kernel=None,
    *,
    t_grid,
    subject_per_row,
    row_bins,
    subjects,
    move_bins,
    padding,
):
    """Mean of a (optionally kernel-convolved) kinematic profile per stimulation bin"""
    feat = np.empty(len(row_bins))
    for s in subjects:
        sig = np.nan_to_num(profile_df[s].values)
        if kernel is not None:
            sig = kernel(sig)
        vals = {}
        for b in TIMEPOINT_ORDER:
            lo, hi = bin_window(b, mt[s], move_bins, padding)
            msk = (t_grid > lo) & (t_grid <= hi)
            vals[b] = sig[msk].mean()
        sel = subject_per_row == s
        feat[sel] = [vals[b] for b in row_bins[sel]]
    return feat


def symmetric_exponential_kernel(t, tau):
    """Symmetric exponential masking kernel, normalized to unit peak"""
    return np.exp(-np.abs(t) / tau)


def asymmetric_exponential_kernel(tau, tau_fwd, tau_bwd):
    """Asymmetric exponential masking kernel (heavier backward tail)"""
    return np.where(tau <= 0, np.exp(tau / tau_fwd), np.exp(-tau / tau_bwd))


def convolve_with_asymmetric_exponential(signal, tau_fwd, tau_bwd, dt):
    """Convolve a signal with the normalized asymmetric exponential kernel"""
    signal = np.nan_to_num(np.asarray(signal, dtype=float))
    W = 6 * max(tau_fwd, tau_bwd)
    tau = np.arange(-W, W + dt, dt)
    k = asymmetric_exponential_kernel(tau, tau_fwd, tau_bwd)
    k = k / k.sum()
    return fftconvolve(signal, k[::-1], mode="same")[: len(signal)]


def convolve_with_symmetric_exponential(signal, tau, dt):
    """Convolve a signal with the normalized symmetric exponential kernel"""
    signal = np.nan_to_num(np.asarray(signal, dtype=float))
    kernel_t = np.arange(-6 * tau, 6 * tau + dt, dt)
    kernel = symmetric_exponential_kernel(kernel_t, tau)
    kernel /= kernel.sum()
    return fftconvolve(signal, kernel, mode="same")[: len(signal)]


def model_stats(pred_all, observed, k_params):
    """R2, MSE, log-likelihood, AIC and BIC of a linear model fit"""
    n = len(pred_all)
    ss = np.sum((observed - pred_all) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    r2 = 1 - ss / ss_tot
    mse = ss / n
    ll = -n / 2 * np.log(2 * np.pi * ss / n) - n / 2
    return r2, mse, ll, -2 * ll + 2 * k_params, -2 * ll + k_params * np.log(n)


def k_bin_window(b, pt_min, pt_max, move_bins):
    """Percent-time window of a stimulation bin for Kalman-gain averaging"""
    if b in move_bins:
        return move_bins[b]
    return {
        "(-inf, -0.1]": (pt_min - 1e-9, -0.1),
        "(-0.1, 0.0]": (-0.1, 0.0),
        "(1.0, inf]": (1.0 - 1e-9, pt_max),
    }[b]


def k_per_bin(data, condition, subject, move_bins):
    """Mean position Kalman gain per stimulation bin for one subject/condition"""
    Ks = data[
        (data["condition"] == condition) & (data["subject"] == subject)
    ].sort_values("percent_time")
    pt, kp = Ks["percent_time"].values, Ks["K_pos"].values
    out = []
    for b in TIMEPOINT_ORDER:
        lo, hi = k_bin_window(b, pt.min(), pt.max(), move_bins)
        msk = (pt > lo) & (pt <= hi)
        out.append(kp[msk].mean())
    return out


def design_columns(condition, profiles, mean_mt, k_gains, feature_fn, kernels):
    """Design matrices (one column set per model) for the linear model fits"""
    mt = mean_mt[condition]
    asym_fn, sym_fn = kernels
    v = feature_fn(profiles[condition]["vel"], mt)  # plain (no kernel)
    a = feature_fn(profiles[condition]["acc"], mt)
    dv_a = feature_fn(profiles[condition]["vel"], mt, asym_fn)
    dv_s = feature_fn(profiles[condition]["vel"], mt, sym_fn)
    da_a = feature_fn(profiles[condition]["acc"], mt, asym_fn)
    da_s = feature_fn(profiles[condition]["acc"], mt, sym_fn)
    k = k_gains[condition]
    return {
        "Vel-only": v[:, None],
        "Acc-only": a[:, None],
        "Linear (v+a)": np.column_stack([v, a]),
        "Conv vel (Asym)": dv_a[:, None],
        "Conv vel (Sym)": dv_s[:, None],
        "Conv acc (Asym)": da_a[:, None],
        "Conv acc (Sym)": da_s[:, None],
        "Conv v+a (Asym)": np.column_stack([dv_a, da_a]),
        "Conv v+a (Sym)": np.column_stack([dv_s, da_s]),
        "OFC K-gain": k[:, None],
    }


def fit_ps_linear(design, d_prime_fit, subjects, subject_per_row):
    """Per-subject-intercept linear fits predicting d' from a design matrix"""
    pred, coefs = np.empty(len(d_prime_fit)), {}
    for s in subjects:
        m = subject_per_row == s
        X = np.column_stack([np.ones(m.sum()), design[m]])
        coefs[s], *_ = lstsq(X, d_prime_fit[m], rcond=None)
        pred[m] = X @ coefs[s]
    return pred, len(subjects) * (design.shape[1] + 1), coefs


def predict_with_coefs(design, coefs, subject_per_row):
    """Predictions from per-subject intercept + slope fits"""
    out = np.empty(len(subject_per_row))
    for s in coefs:
        m = subject_per_row == s
        out[m] = np.column_stack([np.ones(m.sum()), design[m]]) @ coefs[s]
    return out


def bin_mean(rows, row_bins):
    """Mean of per-row values grouped by stimulation time bin"""
    return (
        pd.Series(rows, index=row_bins)
        .groupby(level=0)
        .mean()
        .reindex(TIMEPOINT_ORDER)
        .values
    )


def norm_profiles_signed(reaches_cond, norm_grid, cols):
    """Per-subject signed distance/velocity/acceleration profiles, trial-averaged"""
    prof = {c: {} for c in cols}
    for subject, subj_df in reaches_cond.groupby("subject"):
        curves = {c: [] for c in cols}
        for _, trial in subj_df.groupby(["trial"]):
            trial = trial.sort_values("percent_reach_t")
            x = trial["percent_reach_t"].to_numpy()
            for c in cols:
                curves[c].append(
                    np.interp(
                        norm_grid, x, trial[c].to_numpy(), left=np.nan, right=np.nan
                    )
                )
        for c in cols:
            prof[c][subject] = np.nanmean(curves[c], axis=0)
    return {c: pd.DataFrame(prof[c], index=norm_grid).T for c in cols}


def plot_kinematics(conditions, kin_profiles, norm_grid):
    """Mean +- std distance/velocity/acceleration profiles per condition"""
    fig, axes = plt.subplots(
        1, 3, figsize=(9 * CM_TO_INCHES, 2 * CM_TO_INCHES), sharex=True
    )
    fig.subplots_adjust(wspace=0.7)
    for col, ylab, ax in zip(
        ["d", "v", "a"], ["Distance [m]", "Velocity [m/s]", "Acceler. [m/s²]"], axes
    ):
        for cond in conditions:
            prof = kin_profiles[cond][col]
            m, s = prof.mean(axis=0).values, prof.std(axis=0).values
            ax.plot(
                norm_grid,
                m,
                color=CONDITION_COLORS[cond],
                linestyle=CONDITION_LINESTYLES[cond],
                label=CONDITION_LABELS_DICT[cond],
            )
            ax.fill_between(
                norm_grid, m - s, m + s, color=CONDITION_COLORS[cond], alpha=0.2
            )
        ax.set(ylabel=ylab)
    axes[0].set(ylim=(0, 0.6), xticks=np.arange(0, 1.01, 0.2))
    axes[1].set(xlabel="Time normalized", ylim=(0, 2))
    axes[2].set(ylim=(-10, 15))
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, pad=0.2)
    plt.show()
    return fig, axes
