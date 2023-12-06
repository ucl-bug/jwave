if __name__ == "__main__":
  from jwave.solvers import (
    TimeHarmonicProblem,
    HelmholtzGMRES,
    solve_time_harmonic,
    NormalizeHelmholtz
  )

  from jwave import Medium, FourierSeries, Domain
  from jax import numpy as jnp
  from matplotlib import pyplot as plt
  import time

  domain = Domain(N=(256, 256), dx=(1.0, 1.0))

  sound_speed = jnp.ones(domain.N).at[128:,128:].set(1.2)
  sound_speed = FourierSeries(sound_speed, domain)
  density = jnp.ones(domain.N).at[128:,:128].set(2.)
  density = FourierSeries(density, domain)
  attenuation = jnp.zeros(domain.N).at[:128,128:].set(2.0)
  attenuation = FourierSeries(attenuation, domain)
  medium = Medium(
    domain=domain,
    sound_speed=sound_speed,
    density=density,
    attenuation=attenuation
  )

  source = jnp.zeros(domain.N).at[36, 36].set(1.0) + 0j
  source = FourierSeries(source, domain)
  frequency = 1.0 / (2 * jnp.pi)

  # Define Helmholtz problem
  problem = TimeHarmonicProblem.from_medium(medium, frequency)

  # simple gmres solver
  solution = solve_time_harmonic(
    problem=problem,
    solver=HelmholtzGMRES(maxiter=100),
    source=source,
    processor=NormalizeHelmholtz(),
  )

  # Save field
  field = solution.value.on_grid
  now = time.time()
  plt.imshow(jnp.real(field))
  plt.savefig(f"field_{now}.png")
  plt.close()