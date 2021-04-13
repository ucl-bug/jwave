import os
from jax import numpy as jnp
from jwextras.matlab import Matlab 
import jax
import numpy as np
from matplotlib import pyplot as plt 
from jwave.geometry import kGrid, TimeAxis, Medium, Sources, Sensors
from jwave.signal_processing import apply_ramp
from time import time
from jwave.physics import simulate_wave_propagation

def gaussian_window(signal, t, mu, sigma):
    return signal*jnp.exp(
        -(t-mu)**2/sigma**2
    )

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    N = (512, 512)
    dx = (0.5, 0.5)
    cfl = 0.3
    t_max = 50.

    grid = kGrid.make_grid(N, dx)

    # Physical properties
    sound_speed = jnp.ones(N) 
    # Physical properties
    medium = Medium(
        sound_speed=sound_speed,
        density=jnp.ones(N),
        attenuation=0.0,
        pml_size=30
    )

    time_array = TimeAxis.from_kgrid(grid, medium, cfl=cfl, t_end=t_max)

    # define a source point
    source_freq = 0.2
    source_mag = 5

    t = jnp.arange(0, time_array.t_end, time_array.dt)
    s1 = source_mag * jnp.sin(2 * jnp.pi * source_freq * t)
    s1 = gaussian_window(
        apply_ramp(s1, time_array.dt, source_freq),
        t,
        30,
        10
    )

    source_signals = jnp.stack([s1])
    source_positions = ([50],[50])

    sources = Sources(positions=source_positions, signals=source_signals)


    # Run simulation
    output_taxis = TimeAxis(dt=time_array.t_end, t_end=time_array.t_end)
    fields = simulate_wave_propagation(grid, medium, time_array, sources, backprop=False, output_t_axis=output_taxis)
    u = fields[0]
    p_jwave = jnp.sum(fields[-1],1)*(medium.sound_speed**2)

    sim = jax.jit(
        lambda x: simulate_wave_propagation(
            grid, x, time_array, sources, backprop=False, output_t_axis=output_taxis
        )
    )

    # Compile
    fields = sim(medium)


    # Profile
    jax.profiler.start_server(8892)
    jax.profiler.start_trace("./")

    fields = sim(medium)
    x = fields[0].block_until_ready()

    jax.profiler.stop_trace()


