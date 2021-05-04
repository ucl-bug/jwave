import os
from jax import numpy as jnp
from jwextras.matlab import Matlab 

os.system("module load Matlab/2019a")
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".70"
_ = jnp.array([1.]) # Takes GPU for JAX

os.environ["CUDA_VISIBLE_DEVICES"]="4"
mlb = Matlab()
mlb.start()
mlb.run("addpath('~/repos/jwave/tests/benchmark')")

import jax
import numpy as np
from matplotlib import pyplot as plt 
from jwave.geometry import kGrid, TimeAxis, Medium, Sources, Sensors
from jwave.signal_processing import apply_ramp
from time import time
from jwave.physics import simulate_wave_propagation


# Defining geometry
def gaussian_window(signal, t, mu, sigma):
    return signal*jnp.exp(
        -(t-mu)**2/sigma**2
    )

def single_src_experiment(size=128, pml_size=20, t_max=1000.):
    N = (size, size)
    dx = (0.5, 0.5)
    cfl = 0.3
    t_max = t_max

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
    
    ## Run in k-Wave
    print("k-Wave")
    mlb.add(sound_speed, "sound_speed")
    mlb.add(dx[0], "dx")
    mlb.add([x[0] for x in source_positions], "source_location")
    mlb.add(s1, "source_signal")
    mlb.add(time_array.dt, "dt")
    mlb.add(t_max, "t_max")
    mlb.run("[p, exec_t] = kwave_solver(sound_speed, dx, source_location, source_signal, dt, t_max);")
    kwave_exec_time = mlb.get("exec_t")

    ## Run in j-Wave
    print("j-Wave")
    output_taxis = TimeAxis(dt=time_array.t_end, t_end=time_array.t_end)

    sim = jax.jit(
        lambda x: simulate_wave_propagation(
            grid, x, time_array, sources, backprop=False, output_t_axis=output_taxis
        )
    )
    
    # Compile
    fields = sim(medium)
    y = fields[1].block_until_ready()
    
    # Execute compiled function
    start_time = time()
    fields = sim(medium)
    y = fields[1].block_until_ready()
    jwave_exec_time = time() - start_time
    
    return kwave_exec_time, jwave_exec_time


k_times = []
j_times = []
sizes = list(map(lambda x: 2**x, range(7,12)))

for size in sizes:
    print(size)
    a, b = single_src_experiment(size)
    k_times.append(a)
    j_times.append(b)


plt.figure(figsize=(12,5))
plt.plot(sizes, k_times, label="k-Wave", marker="v")
plt.plot(sizes, j_times, label="j-Wave", marker="o")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Size of the 2D grid edge")
plt.ylabel("Computational time")
plt.xlim([sizes[0], sizes[-1]])
plt.xticks(sizes, list(map(str, sizes)))  # Set text labels.
plt.title("2D domain")
plt.grid()
plt.show()


ratio = list(map(lambda x: x[0]/x[1], zip(j_times, k_times)))

plt.figure(figsize=(12,5))
plt.plot(sizes, ratio, label="Ratio of computation time", marker="v")
plt.legend()
#plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Size of the 2D grid edge")
plt.ylabel("Ratio of execution time jWave/kWave")
plt.xlim([sizes[0], sizes[-1]])
plt.xticks(sizes, list(map(str, sizes)))  # Set text labels.
plt.title("2D domain")
plt.grid()
plt.show()

def single_src_experiment_3d(size=128, pml_size=20, t_max=300.):
    N = (size, size, size)
    dx = (0.5, 0.5, 0.5)
    cfl = 0.3
    t_max = t_max

    grid = kGrid.make_grid(N, dx)
    
    # Physical properties
    sound_speed = jnp.ones(N) 
    # Physical properties
    medium = Medium(
        sound_speed=sound_speed,
        density=jnp.ones(N),
        attenuation=0.0,
        pml_size=20
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
    source_positions = ([25],[25],[25])

    sources = Sources(positions=source_positions, signals=source_signals)
    
    ## Run in k-Wave
    print("k-Wave")
    mlb.add(sound_speed, "sound_speed")
    mlb.add(dx[0], "dx")
    mlb.add([x[0] for x in source_positions], "source_location")
    mlb.add(s1, "source_signal")
    mlb.add(time_array.dt, "dt")
    mlb.add(t_max, "t_max")
    mlb.run("[p, exec_t] = kwave_solver_3D(sound_speed, dx, source_location, source_signal, dt, t_max);")
    kwave_exec_time = mlb.get("exec_t")
    
    ## Run in j-Wave
    print("j-Wave")
    output_taxis = TimeAxis(dt=time_array.t_end, t_end=time_array.t_end)

    sim = jax.jit(
        lambda x: simulate_wave_propagation(
            grid, x, time_array, sources, backprop=False, output_t_axis=output_taxis
        )
    )
    
    # Compile
    fields = sim(medium)
    y = fields[1].block_until_ready()
    
    # Execute compiled function
    start_time = time()
    fields = sim(medium)
    y = fields[1].block_until_ready()
    jwave_exec_time = time() - start_time
    
    return kwave_exec_time, jwave_exec_time


k_times = []
j_times = []
sizes = list(map(lambda x: 2**x, range(6,9)))

for size in sizes:
    print(size)
    a, b = single_src_experiment_3d(size)
    k_times.append(a)
    j_times.append(b)


plt.figure(figsize=(12,5))
plt.plot(sizes, k_times, label="k-Wave", marker="v")
plt.plot(sizes, j_times, label="j-Wave", marker="o")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Size of the 2D grid edge")
plt.ylabel("Computational time")
plt.xlim([sizes[0], sizes[-1]])
plt.xticks(sizes, list(map(str, sizes)))  # Set text labels.
plt.title("3D Domain")
plt.grid()
plt.show()

ratio = list(map(lambda x: x[0]/x[1], zip(j_times, k_times)))

plt.figure(figsize=(12,5))
plt.plot(sizes, ratio, label="Ratio of computation time", marker="v")
plt.legend()
#plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Size of the 2D grid edge")
plt.ylabel("Ratio of execution time jWave/kWave")
plt.xlim([sizes[0], sizes[-1]])
plt.xticks(sizes, list(map(str, sizes)))  # Set text labels.
plt.title("3D Domain")
plt.grid()
plt.show()

# In[23]:


k_times = []
j_times = []
endtime = [200., 300., 400., 500., 750., 1000., 2000., 5000.]

for t_max in endtime:
    print(t_max)
    a, b = single_src_experiment(512, t_max=t_max)
    k_times.append(a)
    j_times.append(b)


plt.figure(figsize=(12,5))
plt.plot(endtime, k_times, label="k-Wave", marker="v")
plt.plot(endtime, j_times, label="j-Wave", marker="o")
plt.legend()
plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Simulation time [sec]")
plt.ylabel("Computational time")
plt.xlim([endtime[0], endtime[-1]])
plt.xticks(endtime, list(map(str, endtime)))  # Set text labels.
plt.title("Domain size: 256x256")
plt.grid()
plt.show()

ratio = list(map(lambda x: x[0]/x[1], zip(j_times, k_times)))

plt.figure(figsize=(12,5))
plt.plot(endtime, ratio, label="Ratio of computation time", marker="v")
plt.legend()
#plt.yscale("log")
plt.xscale("log")
#plt.ylim([0.5,10])
plt.xlabel("Simulation time [sec]")
plt.ylabel("Ratio of execution time jWave/kWave")
plt.xlim([endtime[0], endtime[-1]])
plt.title("3D Domain")
plt.grid()
plt.show()