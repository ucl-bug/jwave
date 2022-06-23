# j-Wave
*Fast and differentiable acoustic simulations in JAX*

j-Wave is a customizable Python simulator, written on top of the [JAX](https://github.com/google/jax) librar and the discretization framework [JaxDF](https://github.com/ucl-bug/jaxdf), for fast, parallelizable, and differentiable acoustic simulations.

j-Wave solves both time-varying and time-harmonic forms of the wave equation with support for multiple discretizations, including finite differences and Fourier spectral methods. Custom discretizations, including those based on neural networks, can also be utilized via the JaxDF framework.

The use of the JAX library gives direct support for program transformations, such as automatic differentiation, Single-Program Multiple-Data (SPMD) parallelism, and just-in-time compilation.

Following the phylosophy of [JAX](https://jax.readthedocs.io/en/stable/), j-Wave is developed with the following principles in mind

1. Fully differentiable
2. Fast via hardware-specific `jit` compilation
3. Easy to run on GPUs and TPUs
4. Easy to customize to support novel research ideas, including novel discretizations via [`jaxdf`](https://github.com/ucl-bug/jaxdf)

[Get started with j-Wave](notebooks/ivp/homogeneous_medium.html){ .md-button }
