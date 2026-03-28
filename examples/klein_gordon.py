"""Klein-Gordon wave equation example.

Solves the Klein-Gordon equation in the frequency domain using the
`klein_gordon` operator provided by j-Wave.

The Klein-Gordon equation for a massive scalar field reads::

    (nabla^2 + omega^2/c^2 - m^2) u = f

where *m* is the field mass.  Setting *m = 0* recovers the ordinary
Helmholtz equation.

This example sets up a 2-D domain with a point source, solves the
equation for two mass values (massless and massive), and plots the
resulting pressure fields side by side.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxdf.discretization import FourierSeries

from jwave.acoustics import klein_gordon
from jwave.geometry import Domain, Medium

# ── Domain setup ──────────────────────────────────────────────────────
N = (128, 128)
dx = (0.1e-3, 0.1e-3)  # 0.1 mm grid spacing
domain = Domain(N, dx)

# Homogeneous medium (water-like)
sound_speed = 1500.0  # m/s
medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=15)

# Angular frequency corresponding to 1 MHz
f0 = 1e6
omega = 2 * jnp.pi * f0

# ── Source term ───────────────────────────────────────────────────────
src = jnp.zeros(N)
src = src.at[N[0] // 2, N[1] // 2].set(1.0)
source = FourierSeries(src, domain)

# ── Solve for two mass values ─────────────────────────────────────────
params = klein_gordon.default_params(source, medium, omega=omega)

# Massless (equivalent to Helmholtz)
kg_massless = klein_gordon(source, medium, omega=omega, mass=0.0, params=params)

# Massive (m = k/2, where k = omega / c)
k0 = omega / sound_speed
mass = k0 / 2
kg_massive = klein_gordon(source, medium, omega=omega, mass=mass, params=params)

# ── Visualisation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, field, label in zip(
    axes,
    [kg_massless, kg_massive],
    ["Helmholtz (m = 0)", f"Klein-Gordon (m = k/2)"],
):
    img = jnp.abs(field.on_grid[..., 0])
    ax.imshow(img.T, cmap="inferno", origin="lower")
    ax.set_title(label)
    ax.set_xlabel("x [grid points]")
    ax.set_ylabel("y [grid points]")

plt.tight_layout()
plt.savefig("klein_gordon_example.png", dpi=150)
plt.show()
print("Done.")
