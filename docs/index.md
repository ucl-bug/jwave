# jWave Documentation

jWave is a library for performing pseudospectral simulations of acoustic signals. Is heavily inspired by [k-Wave](http://www.k-wave.org/) (in its essence, is a port of k-Wave in JAX), and its intented to be used as a collection of modular blocks that can be easily included into any machine learning pipeline.

Following the phylosophy of [JAX](https://jax.readthedocs.io/en/stable/), jWave is developed with the following principles in mind

1. Differntiable
2. Fast via `jit` compilation
3. Easy to run on GPUs