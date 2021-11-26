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
from jwave.acoustics import ongrid_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask
from jwave.phantoms import three_circles
from jax import numpy as jnp
from jax import jit, grad

# Simulation parameters
domain = Domain(N=(128, 128), dx=(0.1e-3, 0.1e-3))
medium = Medium(domain=domain, sound_speed=jnp.ones(N)*1500)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=.8e-05)
x, y = _points_on_circle(32,40,(64,64))  # Place sensors on a circle
sensors = Sensors(positions=(jnp.array(x), jnp.array(y)))

# Initial pressure distribution
p0 = three_circles(domain.N)

# Construct differentiable solver
from jwave.acoustics import ongrid_wave_propagation

params, solver = ongrid_wave_propagation(
    medium=medium,
    time_array=time_axis,
    output_t_axis = time_axis,
    sensors=sensors,
    backprop=True,
    p0 = p0
)

# Compile and run simulation
sensors_data = jit(solver)(params) 

# Make imaging algorithm
@jit 
def lazy_time_reversal(p):
    def mse_loss(p0):
        local_params = params.copy()
        local_params["initial_fields"]["p"] = p0
        p_pred = solver(local_params)
        return 0.5*jnp.sum(jnp.abs(p_pred - p)**2)
    p0 = jnp.zeros_like(params["initial_fields"]["p"])
    return - grad(mse_loss)(p0)

# Reconstruct image from noisy data
noise = random.normal(random.PRNGKey(42), sensors_data.shape)
for i in range(noise.shape[1]):
    noise = noise.at[:,i].set(smooth(noise[:,i]))
noisy_data = sensors_data + 0.5*noise

recon_image = lazy_time_reversal(noisy_data)
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
