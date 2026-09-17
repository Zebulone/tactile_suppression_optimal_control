from jax import numpy as jnp, vmap
from jax.lax import scan
from jax import jit, random

from .util import quadratic_form, quadratic_form_t


@jit
def backward_pass(lqg, K, x0, Sigma0):
    T = lqg.A.shape[0]
    xdim = lqg.A.shape[1]

    def lqr_iter(carry, t):
        Sx, Se, s = carry
        A, B, V, C, H, W, D, E = (
            lqg.A[t],
            lqg.B[t],
            lqg.V[t],
            lqg.C[t],
            lqg.H[t],
            lqg.W[t],
            lqg.D[t],
            lqg.E[t],
        )
        Q, R = lqg.Q[t], lqg.R[t]

        L = (
            jnp.linalg.inv(
                R + B.T @ Sx @ B + quadratic_form(C, B.T @ (Sx + Se) @ B).sum(axis=0)
            )
            @ B.T
            @ Sx
            @ A
        )

        s_new = (
            jnp.trace(Sx @ V @ V.T + Se @ (V @ V.T + E @ E.T + K[t] @ W @ W.T @ K[t].T))
            + s
        )
        Sx_new = (
            Q
            + A.T @ Sx @ (A - B @ L)
            + quadratic_form(D, K[t].T @ Se @ K[t]).sum(axis=0)
        )
        Se_new = A.T @ Sx @ B @ L + (A - K[t] @ H).T @ Se @ (A - K[t] @ H)

        return (Sx_new, Se_new, s_new), L

    (Sx, Se, s), L = scan(
        lqr_iter, (lqg.Q[-1], jnp.zeros((xdim, xdim)), 0.0), jnp.arange(T), reverse=True
    )

    cost = x0.T @ Sx @ x0 + jnp.trace((Sx + Se) @ Sigma0) + s

    return L, cost


@jit
def forward_pass(lqg, L, x0, Sigma0):
    T = lqg.A.shape[0]
    xdim = lqg.A.shape[1]

    def kalman_iter(carry, t):
        Se, Sx, Sxe = carry
        A, B, V, C, H, W, D, E = (
            lqg.A[t],
            lqg.B[t],
            lqg.V[t],
            lqg.C[t],
            lqg.H[t],
            lqg.W[t],
            lqg.D[t],
            lqg.E[t],
        )

        G = (
            H @ Se @ H.T
            + W @ W.T
            + quadratic_form_t(D, (Se + Sx + Sxe + Sxe.T)).sum(axis=0)
        )
        F = A @ Se @ H.T
        K = A @ Se @ H.T @ jnp.linalg.inv(G)

        Se = (
            V @ V.T
            + E @ E.T
            + (A - K @ H) @ Se @ A.T
            + B @ quadratic_form_t(C, L[t] @ Sx @ L[t].T).sum(axis=0) @ B.T
        )
        AmBL = A - B @ L[t]
        Sx = (
            E @ E.T
            + K @ H @ Se @ A.T
            + AmBL @ Sx @ AmBL.T
            + AmBL @ Sxe @ H.T @ K.T
            + K @ H @ Sxe.T @ AmBL.T
        )
        Sxe = AmBL @ Sxe @ (A - K @ H).T - E @ E.T

        return (Se, Sx, Sxe), (G, K, F)

    _, (G, K, F) = scan(
        kalman_iter, (Sigma0, jnp.outer(x0, x0), jnp.zeros((xdim, xdim))), jnp.arange(T)
    )

    return G, K, F


def solve(lqgspec, x0, Sigma0=None, max_iter=10):
    T = lqgspec.A.shape[0]

    xdim = lqgspec.A.shape[1]
    ydim = lqgspec.H.shape[1]
    udim = lqgspec.B.shape[2]

    if Sigma0 is None:
        Sigma0 = lqgspec.V[0] @ lqgspec.V[0].T

    G = jnp.zeros((T, ydim, ydim))
    F = jnp.zeros((T, xdim, ydim))
    K = jnp.zeros((T, xdim, ydim))
    L = jnp.zeros((T, udim, xdim))

    def solver_loop(x, i):
        _, K, L, _ = x
        L, cost = backward_pass(lqgspec, K, x0, Sigma0)
        G, K, F = forward_pass(lqgspec, L, x0, Sigma0)

        return (G, K, L, F), cost

    (G, K, L, F), costs = scan(solver_loop, (G, K, L, F), jnp.arange(max_iter))

    return G, K, L, F, costs


def simulate_single(key, lqg, K, L, x0, xhat0=None, average=False):
    xhat0 = xhat0 if xhat0 is not None else x0
    T = lqg.A.shape[0]
    xdim = lqg.A.shape[1]
    ydim = lqg.H.shape[1]

    if average:
        state_noise = jnp.zeros((T, xdim))
        obs_noise = jnp.zeros((T, ydim))
        c_noise = jnp.zeros((T, lqg.C.shape[-2]))
        d_noise = jnp.zeros((T, lqg.D.shape[-2]))
        e_noise = jnp.zeros((T, xdim))
    else:
        key, key_system_noise, key_obs_noise, key_c_noise, key_d_noise, key_e_noise = (
            random.split(key, 6)
        )

        state_noise = random.normal(key_system_noise, (T, xdim))
        obs_noise = random.normal(key_obs_noise, (T, ydim))

        c_noise = random.normal(key_c_noise, (T, lqg.C.shape[-2]))
        d_noise = random.normal(key_d_noise, (T, lqg.D.shape[-2]))
        e_noise = random.normal(key_e_noise, (T, xdim))

    def sim_iter(carry, t):
        x, xhat = carry

        y = lqg.H[t] @ x + lqg.W[t] @ obs_noise[t] + lqg.D[t] @ x @ d_noise[t]

        u = -L[t] @ xhat

        x = (
            lqg.A[t] @ x
            + lqg.B[t] @ u
            + lqg.V[t] @ state_noise[t]
            + lqg.B[t] @ (lqg.C[t] @ u @ c_noise[t])
        )

        xpred = lqg.A[t] @ xhat - lqg.B[t] @ L[t] @ xhat
        inov = y - lqg.H[t] @ xhat
        xhat = xpred + K[t] @ inov + lqg.E[t] @ e_noise[t]

        return (x, xhat), (x, inov)

    _, x = scan(sim_iter, (x0, xhat0), jnp.arange(T))

    return x


def simulate(rng_key, n, model, K, L, x0, xhat0=None):
    rng_keys = random.split(rng_key, n)
    x = vmap(lambda key: simulate_single(key, model, K, L, x0=x0, xhat0=xhat0))(
        rng_keys
    )

    return x


def simulate_different_start(
    rng_key, n, model, K, L, x0, start_offset, delay=11, xhat0=None
):

    rng_key, rng_start = random.split(rng_key)
    random_starts = random.normal(rng_start, shape=(n,)) * start_offset

    rng_keys = random.split(rng_key, n)

    if delay == 0:
        delay = 1

    x = vmap(
        lambda key, start: simulate_single(
            key,
            model,
            K,
            L,
            x0=x0 + jnp.array(delay * [start, 0, 0, 0]),
            xhat0=xhat0,
        )
    )(rng_keys, random_starts)

    return x
