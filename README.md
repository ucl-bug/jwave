# jWave

[![codecov](https://codecov.io/gh/astanziola/jwave/branch/main/graph/badge.svg?token=6J03OMVJS1)](https://codecov.io/gh/astanziola/jwave)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

jWave is a library of simulators for acoustic applications. Is heavily inspired by [k-Wave](http://www.k-wave.org/) (a big portion of `jwave` is a port of k-Wave in JAX), and its intented to be used as a collection of modular blocks that can be easily included into any machine learning pipeline.

Following the phylosophy of [JAX](https://jax.readthedocs.io/en/stable/), jWave is developed with the following principles in mind

1. Differntiable
2. Fast via `jit` compilation
3. Easy to run on GPUs


## :floppy_disk: Install
Before installing `jwave`, make sure that [you have installed `jaxdf`](https://github.com/google/jax#installation). 

Install jwave by `cd` in the repo folder an run
```bash
pip install -r requirements.txt
pip install -e .
```

If you want to run the notebooks, you should also install the following packages
```bash
pip install jupyter, tqdm
```

## Related

### Projects

1. [`ADSeismic.jl`](https://github.com/kailaix/ADSeismic.jl): a finite difference acoustic simulator with support for AD and JIT compilation in Julia.
2. [`stride`](https://github.com/trustimaging/stride): a general optimisation framework for medical ultrasound tomography.
