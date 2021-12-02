# j-Wave extras

This sub-module is a collection of tools to extend the functionality of `jwave`, but that are not part of the main `jwave` package. The tools are not intended to be used by the end user, but are rather for developers.

## MATLAB Engine
`jwave` is still in active development, and a lot of functionality is missing compared to `kWave`, expecially in terms of functions constructing objects. 

If you want to use some `kWave` methods from python (for example the `kArray` class implementing off-grid sources), you should first install the MATLAB engine for python. For example:

```bash
module load Matlab/2019a
cd /apps/software/Matlab/R2019a/extern/engines/python/
python setup.py build --build-base=$(mktemp -d) install
```

Then you can start a new MATLAB session in python as follows
```
from jwave.extras.matlab import Matlab

mlb = Matlab()
mlb.start()
```

If you already have an opened MATLAB session [with shared engine](https://it.mathworks.com/help/matlab/ref/matlab.engine.shareengine.html), then `mlb.start()` will join the session and the its MATLAB workspace will be updated as the python script proceeds. This is quite useful for debugging, or for passing and checking data in MATLAB itself.

Look at the example [notebook to learn more](examples/include_matlab_computations.ipynb).

## kWaveSolver

An interface for solving the wave equation using the MATLAB version of `kWaveFirstOrder`. Useful for comparing simulations against `kWave` results.

**Example:**

```python
from jwave.extras.external_solvers import kWaveSolver

N, dx = (256, 256), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)
sound_speed = jnp.ones(N)*1500
medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=20)
time_axis = TimeAxis.from_medium(medium, cfl=0.3)

sensors_positions = (jnp.array([35]), jnp.array([40]))
sensors = Sensors(positions=sensors_positions)
p0 = 5.*_circ_mask(N, 8, (100,100))

params, _j_solver_ = ongrid_wave_propagation(
    medium=medium,
    time_array=time_axis,
    output_t_axis = time_axis,
    sensors=sensors,
    backprop=False,
    p0 = p0
)

k_solver = kWaveSolver()
p_kwave, kwave_time = k_solver.solve(
    params, 
    dx, 
    time_axis.t_end,
    sensors=sensors,
    p0=p0
)
```

<br/>

## `jwave.extras`

## ::: jwave.extras
    handler: python
    members:
        - engine
        - external_solvers
    show_root_heading: true
    show_source: false