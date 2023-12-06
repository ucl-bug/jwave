if __name__ == "__main__":
  from jwave.solvers import (
    TimeHarmonicProblem,
    HelmholtzGMRES,
    solve_time_harmonic,
    OsnabruggeBC,
    BornSeries
  )

  from jwave import Medium, FourierSeries, Domain
  from jax import numpy as jnp
  from matplotlib import pyplot as plt
  import time
  from jaxdf.operators import gradient, diag_jacobian, sum_over_dims
  import jax
  import equinox as eqx
  
  domain = Domain(N=(256, 256), dx=(1.0, 1.0))
  
  sound_speed = jnp.ones(domain.N).at[128:,128:].set(1.2)
  sound_speed = FourierSeries(sound_speed, domain)
  density = jnp.ones(domain.N).at[128:192,64:128].set(2.)
  density = FourierSeries(density, domain)
  medium = Medium(
    domain=domain,
    sound_speed=sound_speed,
    density=density,
    attenuation=0.0,
    pml_size=0.0
  )
  
  source = jnp.zeros(domain.N).at[36, 36].set(1.0) + 0j
  source = FourierSeries(source, domain)
  frequency = 1.0 / (2 * jnp.pi)
  
  # Define Helmholtz problem
  problem = TimeHarmonicProblem.from_medium(medium, frequency)
  guess = source * 0.0
  
  # Add an outer absorption layer according to Osnabrugge et al.
  problem, souurce, guess, _ = OsnabruggeBC(alpha=1.0, N=10).preprocess(
    problem, source, guess
  )
  
  # Define constants (could be optimized)
  a = 1.0
  b = 1.0
  c = 0.0 # This needs to be zero because we are using the modified operators
  
  # Define the fields
  rho0 = problem.density
  c0 = problem.sound_speed
  k2 = (problem.omega ** 2)/( rho0 * (c0 ** 2) )
  
  # Define the elements of the scattering potential
  V_11 = rho0 - a
  V_22 = k2 - b - 1j*c
  
  # Finding constant to ensure convergence
  c = 0.95 * jnp.max([
    jnp.max(jnp.abs(V_11)),
    jnp.max(jnp.abs(V_22))
  ])
  
  # Construct the guess fields
  u0 = FourierSeries(jnp.zeros(domain.N + (2,)),domain)
  p0 = guess
  
  class FullVec(eqx.Module):
    u: FourierSeries
    p: FourierSeries
    
    def __add__(self, other):
      return FullVec(u=self.u + other.u, v=self.p + other.p)
    
    def __sub__(self, other):
      return FullVec(u=self.u - other.u, v=self.p - other.p)
    
    def __mul__(self, other):
      return FullVec(u=self.u * other, v=self.p * other)
    
    def __rmul__(self, other):
      return self.__mul__(other)
    
  class AugmentedVec(eqx.Module):
    x: FullVec
    x_bar: FullVec
    
    def __add__(self, other):
      return AugmentedVec(x=self.x + other.x, x_bar=self.x_bar + other.x_bar)
    
    def __sub__(self, other):
      return AugmentedVec(x=self.x - other.x, x_bar=self.x_bar - other.x_bar)
    
    def __mul__(self, other):
      return AugmentedVec(x=self.x * other, x_bar=self.x_bar * other)
    
    def __rmul__(self, other):
      return self.__mul__(other)
    
  v0 = FullVec(u=u0, p=p0)
  
  # Construct the base operators  
  def grad_dot(u):
    return sum_over_dims(diag_jacobian(u))
  
  def L0(v):
    u, p = v.u, v.v
    
    u1 = a * u + 1j * gradient(p)
    p1 = 1j * grad_dot(u) + (b + 1j*c) * p

    return FullVec(u = u1, p = p1)
  
  def V0(v):
    u, p = v
    
    return FullVec(u = V_11 * u, p = V_22 * p)
  
  def B0(v):
    u, p = v
    u1, p1 = V0(v)
    return FullVec(u = u - u1, p = p - p1)

  # Construct the adjoint operators using autodiff
  # Note that this could be evaluated in one go with the primals, now
  # we are throwing away the primal values
  L0_adj = jax.vjp(L0, v0)[1]
  V0_adj = jax.vjp(V0, v0)[1]
  B0_adj = jax.vjp(B0, v0)[1]
  
  # Construct the skew-symmetric operators
  x = AugmentedVec(x=v0, x_bar=v0)  
  
  inv_c = 1.0 / c
  
  def L(x):
    v, w = x
    return (-inv_c * L0_adj(w), inv_c * L0(v))
  
  def V(x):
    v, w = x
    return (-inv_c * V0_adj(w), inv_c * V0(v))
  
  def B(x):
    v, w = x
    return (-inv_c * B0_adj(w), inv_c * B0(v))
  
  # Define the single term of the iterative series
  alpha = 0.5
  
  def single_iteration(x, y):
    Bx_y = B(x) + y
    
    Δ = B(
      
    )
    