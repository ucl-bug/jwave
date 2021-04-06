from jwave import geometry, signal_processing, physics
from jwave.utils import assert_pytree_isclose
from jax import numpy as jnp
from jax import grad
from jax import jit, disable_jit


def test_if_helmholtz_problem_runs():
    # Defining geometry

    N = (64, 64)
    dx = (1.0, 1.0)
    omega = 1.0

    # Making geometry
    grid = geometry.kGrid.make_grid(N, dx)

    # Physical properties
    sound_speed = jnp.zeros(N)
    sound_speed = sound_speed.at[32:48, 32:48].set(1.0)

    attenuation = jnp.zeros(N)
    attenuation = attenuation.at[16:32, 16:32].set(0.1)

    density = jnp.ones(N)
    density = density.at[16:32:, 32:48].set(1.5)

    medium = geometry.Medium(
        sound_speed=sound_speed, density=density, attenuation=attenuation, pml_size=12
    )

    # Source field
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[48, 16].set(1.0)

    # Solve
    _ = physics.solve_helmholtz(
        grid, medium, src_field, omega, method="gmres", maxiter=50
    ).block_until_ready()
    _ = physics.solve_helmholtz(
        grid, medium, src_field, omega, method="bicgstab", maxiter=50
    ).block_until_ready()


def test_if_simple_problem_runs():
    N = (32, 32)
    dx = (0.5, 0.5)
    cfl = 0.1

    grid = geometry.kGrid.make_grid(N, dx)

    medium = geometry.Medium(
        sound_speed=jnp.ones(N), density=jnp.ones(N), attenuation=0.0, pml_size=5.0
    )

    time_array = geometry.TimeAxis.from_kgrid(grid, medium, cfl=cfl, t_end=1.0)

    # define a source point
    source_freq = 0.1  # [Hz]
    source_mag = 5
    t = jnp.arange(0, time_array.t_end, time_array.dt)
    s = source_mag * jnp.sin(2 * jnp.pi * source_freq * t)
    s = signal_processing.apply_ramp(s, time_array.dt, source_freq)
    source_signals = jnp.stack([s])
    source_positions = ([12], [12])
    sources = geometry.Sources(positions=source_positions, signals=source_signals)

    _ = physics.simulate_wave_propagation(grid, medium, time_array, sources)


def test_agreement_wave_forward_models():
    disable_jit()
    N = (128, 128)
    dx = (0.5, 0.5)
    cfl = 0.3

    grid = geometry.kGrid.make_grid(N, dx)

    # Physical properties
    sound_speed = jnp.ones(N)

    # Physical properties
    medium = geometry.Medium(
        sound_speed=sound_speed, density=jnp.ones(N), attenuation=0.0, pml_size=20
    )

    time_array = geometry.TimeAxis.from_kgrid(grid, medium, cfl=cfl, t_end=70.0)

    # define a source point
    from jwave.signal_processing import apply_ramp

    source_freq = 0.5
    source_mag = 5 / time_array.dt

    t = jnp.arange(0, time_array.t_end, time_array.dt)
    s1 = source_mag * jnp.sin(2 * jnp.pi * source_freq * t)
    s1 = signal_processing.gaussian_window(
        apply_ramp(s1, time_array.dt, source_freq), t, 10, 3
    )
    source_signals = jnp.stack([s1])
    source_positions = ([32], [32])
    sources = geometry.Sources(positions=source_positions, signals=source_signals)

    # Define sensors
    sensors_positions = ([96], [96])
    sensors = geometry.Sensors(positions=sensors_positions)

    # Run the simulation on the backward version;
    fields = physics.simulate_wave_propagation(
        grid, medium, time_array, sources, sensors, backprop=True
    )
    p_bwd = jnp.sum(fields[1], 1) * (medium.sound_speed[sources.positions] ** 2)

    # Run simulation on euler integrator without vjp
    fields = physics.simulate_wave_propagation(
        grid, medium, time_array, sources, sensors, backprop=False
    )

    p_fwd = jnp.sum(fields[1], 1) * (medium.sound_speed[sources.positions] ** 2)

    # TODO: Those precisions need to be set too high, figure out why (computational graph
    # should be the same)
    assert_pytree_isclose(p_fwd, p_bwd, relative_precision=1e-1, abs_precision=2e-1)


def test_backprop_in_wave_equation_for_nans():

    N = (64, 64)
    dx = (0.5, 0.5)

    grid = geometry.kGrid.make_grid(N, dx)
    time_array = geometry.TimeAxis(t_end=10, dt=0.2)

    source_freq = 0.3
    source_mag = 5

    t = jnp.arange(0, time_array.t_end, time_array.dt)
    s = source_mag * jnp.sin(2 * jnp.pi * source_freq * t)
    s = signal_processing.apply_ramp(s, time_array.dt, source_freq)
    source_signals = jnp.stack([s])
    source_positions = ([16], [16])

    sources = geometry.Sources(positions=source_positions, signals=source_signals)

    def loss(sos):
        medium = geometry.Medium(
            sound_speed=sos, density=1.0, attenuation=0.0, pml_size=12
        )
        fields = physics.simulate_wave_propagation(
            grid, medium, time_array, sources, backprop=True
        )
        p = jnp.sum(fields[1], 1)
        return jnp.mean(jnp.abs(p[:, 48, 48]))

    gradient = jit(grad(loss))(jnp.ones(N))
    assert not jnp.any(jnp.isnan(gradient))


if __name__ == "__main__":
    test_agreement_wave_forward_models()
    test_if_simple_problem_runs()
    test_backprop_in_wave_equation_for_nans()
    test_if_helmholtz_problem_runs()
