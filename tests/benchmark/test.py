import jax
import jax.profiler
import jax.numpy as jnp
import jax.random
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

jax.profiler.start_server(1234)

_ = input("waiting for signal to continue")

key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (1000,1000))
b = jnp.matmul(a, a)