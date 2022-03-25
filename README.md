# j-Wave

[![codecov](https://codecov.io/gh/astanziola/jwave/branch/main/graph/badge.svg?token=6J03OMVJS1)](https://codecov.io/gh/astanziola/jwave)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

j-Wave is a library of simulators for acoustic applications. Is heavily inspired by [k-Wave](http://www.k-wave.org/) (a big portion of `jwave` is a port of k-Wave in JAX), and its intented to be used as a collection of modular blocks that can be easily included into any machine learning pipeline.

Following the phylosophy of [JAX](https://jax.readthedocs.io/en/stable/), j-Wave is developed with the following principles in mind

1. Differntiable
2. Fast via `jit` compilation
3. Easy to run on GPUs


## Example

This example simulates an acoustic initial value problem, which is often used as a simple model for photoacoustic imaging, and uses autodiff to build a simple, stable, time-reversal imaging algorithm:

```python
from jax import jit
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask

# Simulation parameters
domain = Domain(N=(256, 256), dx=(0.1e-3, 0.1e-3))
medium = Medium(domain=domain, sound_speed=1500.)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=.8e-05)

# Initial pressure field
p0 = _circ_mask(domain.N, domain.N[0] / 32, (domain.N[0] / 2, domain.N[1] / 2))
p0 = p0 + _circ_mask(domain.N, domain.N[0] / 42, (domain.N[0] / 3, domain.N[1] / 4))
p0 = FourierSeries(jnp.expand_dims(p0,-1), domain)

# Compile and run the simulation
@jit
def solver(medium, p0):
  return simulate_wave_propagation(medium, time_axis, p0=p0)

pressure = solver(medium, p0)
```

![Reconstructed image using autograd](docs/assets/images/readme_example_reconimage.png)

## :floppy_disk: Install
Before installing `jwave`, make sure that [you have installed `jaxdf`](https://github.com/ucl-bug/jaxdf).

Install jwave by `cd` in the repo folder an run
```bash
pip install -r requirements.txt
pip install -e .
```

### Related Projects

1. [`ADSeismic.jl`](https://github.com/kailaix/ADSeismic.jl): a finite difference acoustic simulator with support for AD and JIT compilation in Julia.
2. [`stride`](https://github.com/trustimaging/stride): a general optimisation framework for medical ultrasound tomography.
