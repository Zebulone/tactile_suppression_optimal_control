import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from typing import Tuple

from .problem import LQGSpec

# Constants for default values
DEFAULT_SIGMA_POS = 0.04
DEFAULT_SIGMA_VEL = 0.14
DEFAULT_SIGMA_FRC = 0.7
DEFAULT_C = 0.7
DEFAULT_R = 1e-5
DEFAULT_E = 0.0
DEFAULT_T = 91
DEFAULT_DT = 0.01
DEFAULT_V = 0.04
DEFAULT_F = 0.4
DEFAULT_OBSERVED_DIMS = jnp.array((0, 1, 2))
DEFAULT_XI = 0.0


def reaching_task_matrices(
    sigma_pos: float = DEFAULT_SIGMA_POS,
    sigma_vel: float = DEFAULT_SIGMA_VEL,
    sigma_frc: float = DEFAULT_SIGMA_FRC,
    c: float = DEFAULT_C,
    r: float = DEFAULT_R,
    e: float = DEFAULT_E,
    T: int = DEFAULT_T,
    dt: float = DEFAULT_DT,
    v: float = DEFAULT_V,
    f: float = DEFAULT_F,
    observed_dims: jnp.ndarray = DEFAULT_OBSERVED_DIMS,
    xi: float = DEFAULT_XI,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Generate matrices for the reaching task.

    Args:
        sigma_pos (float): Perceptual uncertainty on position.
        sigma_vel (float): Perceptual uncertainty on velocity.
        sigma_frc (float): Perceptual uncertainty on force.
        c (float): Control-dependent noise.
        r (float): Cost of actions.
        e (float): Estimation noise.
        T (int): Number of time steps.
        dt (float): Duration of time steps in seconds.
        v (float): Endpoint velocity penalty.
        f (float): Endpoint force penalty.
        observed_dims (jnp.ndarray): Observed dimensions.
        xi (float): Control noise.

    Returns:
        tuple: Matrices A, B, V, C, H, W, D, E, Q_T, R.
    """
    # Setup problem
    m: float = 1  # mass(kg)
    b: float = 0  # damping(N / sec)
    tau: float = 40  # time constant(msec)

    # Compute system dynamics and cost matrices
    dtt: float = dt / (tau / 1000)
    A: jnp.ndarray = jnp.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1 - dt * b / m, dt / m, 0.0],
            [0.0, 0.0, 1 - dtt, dtt],
            [0.0, 0.0, 0.0, 1 - dtt],
        ]
    )

    B: jnp.ndarray = jnp.array([[0.0], [0.0], [0.0], [dtt]])

    # Control noise
    C: jnp.ndarray = jnp.array([[[c]]])  # Signal dependent
    V: jnp.ndarray = jnp.diag(jnp.array([1, 1, 1, 0])) * xi  # Normal

    # Observation matrix
    H: jnp.ndarray = jnp.eye(4).take(observed_dims, axis=0)

    # Observation noise
    D: jnp.ndarray = jnp.stack(
        (
            jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0])),
            jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0])),
            jnp.diag(jnp.array([0.0, 0.0, 0.0, 0.0])),
        ),
        axis=1,
    ).take(
        observed_dims, axis=0
    )  # Signal dependent

    W: jnp.ndarray = jnp.diag(
        jnp.array([sigma_pos, sigma_vel, sigma_frc, sigma_frc]).take(observed_dims)
    )  # Normal

    # State noise
    E: jnp.ndarray = jnp.diag(jnp.array([1, 1, 1, 1])) * e

    # Control penalty
    R: jnp.ndarray = jnp.array([[r / T]])

    # Final time step costs
    d: jnp.ndarray = jnp.diag(jnp.array([1, v, f, 0]))
    Q: jnp.ndarray = d.T @ d

    return A, B, V, C, H, W, D, E, Q, R


def make_reaching_task(
    sigma_pos: float = DEFAULT_SIGMA_POS,
    sigma_vel: float = DEFAULT_SIGMA_VEL,
    sigma_frc: float = DEFAULT_SIGMA_FRC,
    c: float = DEFAULT_C,
    r: float = DEFAULT_R,
    e: float = DEFAULT_E,
    T: int = DEFAULT_T,
    dt: float = DEFAULT_DT,
    v: float = DEFAULT_V,
    f: float = DEFAULT_F,
    observed_dims: jnp.ndarray = DEFAULT_OBSERVED_DIMS,
) -> LQGSpec:
    """
    Create a reaching task specification.

    Args:
        sigma_pos (float): Perceptual uncertainty on position.
        sigma_vel (float): Perceptual uncertainty on velocity.
        sigma_frc (float): Perceptual uncertainty on force.
        c (float): Control-dependent noise.
        r (float): Cost of actions.
        e (float): Estimation noise.
        T (int): Number of time steps.
        dt (float): Duration of time steps in seconds.
        v (float): Endpoint velocity penalty.
        f (float): Endpoint force penalty.
        observed_dims (jnp.ndarray): Observed dimensions.

    Returns:
        LQGSpec: Linear Quadratic Gaussian specification for the task.
    """
    A, B, V, C, H, W, D, E, Qt, R = reaching_task_matrices(
        sigma_pos,
        sigma_vel,
        sigma_frc,
        c,
        r,
        e,
        T,
        dt,
        v=v,
        f=f,
        observed_dims=observed_dims,
    )
    Q: jnp.ndarray = jnp.zeros((T, 5, 5))
    Q = Q.at[-1].set(Qt)

    return LQGSpec(
        A=jnp.stack([A] * (T - 1)),
        B=jnp.stack([B] * (T - 1)),
        V=jnp.stack([V] * (T - 1)),
        C=jnp.stack([C] * (T - 1)),
        H=jnp.stack([H] * (T - 1)),
        W=jnp.stack([W] * (T - 1)),
        D=jnp.stack([D] * (T - 1)),
        E=jnp.stack([E] * (T - 1)),
        Q=Q,
        R=jnp.stack([R] * (T - 1)),
    )


def make_delayed_reaching_task_plus(
    delay=0,
    sigma_pos=DEFAULT_SIGMA_POS,
    sigma_vel=DEFAULT_SIGMA_VEL,
    sigma_frc=DEFAULT_SIGMA_FRC,
    c=DEFAULT_C,
    r=DEFAULT_R,
    e=DEFAULT_E,
    xi=DEFAULT_XI,
    T=DEFAULT_T,
    dt=DEFAULT_DT,
    v=DEFAULT_V,
    f=DEFAULT_F,
    observed_dims=DEFAULT_OBSERVED_DIMS,
    initial_period=20,
    final_period=1,
):
    """
    Create a delayed reaching task specification with additional periods.

    Args:
        delay (int): Delay in time steps.
        sigma_pos (float): Perceptual uncertainty on position.
        sigma_vel (float): Perceptual uncertainty on velocity.
        sigma_frc (float): Perceptual uncertainty on force.
        c (float): Control-dependent noise.
        r (float): Cost of actions.
        e (float): Estimation noise.
        xi (float): Additional noise parameter.
        T (int): Number of time steps.
        dt (float): Duration of time steps in seconds.
        v (float): Endpoint velocity penalty.
        f (float): Endpoint force penalty.
        observed_dims (array): Observed dimensions.
        initial_period (int): Duration of initial period.
        final_period (int): Duration of final period.

    Returns:
        LQGSpec: Linear Quadratic Gaussian specification for the delayed task.
    """
    A, B, V, C, H, W, D, E, Qt, R = reaching_task_matrices(
        sigma_pos,
        sigma_vel,
        sigma_frc,
        c,
        r,
        e,
        T,
        dt,
        xi=xi,
        v=v,
        f=f,
        observed_dims=observed_dims,
    )

    d = A.shape[0]

    A = jlinalg.block_diag(A, jnp.diag(jnp.zeros(d * delay))) + jnp.diag(
        jnp.ones(d * delay), k=-d
    )

    B = jnp.vstack([B] + [jnp.zeros_like(B)] * delay)
    V = jlinalg.block_diag(V, jnp.diag(jnp.zeros(d * delay)))

    H = jnp.hstack([jnp.zeros((H.shape[0], H.shape[1] * delay)), H])

    D = jnp.concatenate(
        [jnp.zeros((D.shape[0], D.shape[1], D.shape[2] * delay)), D], axis=-1
    )

    E = jlinalg.block_diag(*[E] * (delay + 1))

    Q = jnp.zeros((T, A.shape[0], A.shape[0]))

    # Initial cost to stay at starting position
    Q0 = jnp.zeros((Qt.shape[0], Qt.shape[1]))
    Q0 = Q0.at[0, 0].set(1)
    Q0 = Q0.at[1, 1].set(1)
    Q0 = Q0.at[2, 2].set(1)
    Q0 = jlinalg.block_diag(Q0, *[jnp.zeros_like(Qt)] * delay)
    Q0 = jnp.tile(Q0, (initial_period, 1, 1))
    Q = Q.at[:initial_period].set(Q0)

    # Final cost to stay at target
    Qt = jlinalg.block_diag(Qt, *[jnp.zeros_like(Qt)] * delay)
    Qt = jnp.tile(Qt, (final_period, 1, 1))
    Q = Q.at[-final_period:].set(Qt)
    return LQGSpec(
        A=jnp.stack([A] * (T - 1)),
        B=jnp.stack([B] * (T - 1)),
        V=jnp.stack([V] * (T - 1)),
        C=jnp.stack([C] * (T - 1)),
        H=jnp.stack([H] * (T - 1)),
        W=jnp.stack([W] * (T - 1)),
        D=jnp.stack([D] * (T - 1)),
        E=jnp.stack([E] * (T - 1)),
        Q=Q,
        R=jnp.stack([R] * (T - 1)),
    )
