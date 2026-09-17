from jax import vmap

quadratic_form = vmap(lambda A, B: A.T @ B @ A, in_axes=(1, None))

quadratic_form_t = vmap(lambda A, B: A @ B @ A.T, in_axes=(1, None))