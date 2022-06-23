# `jwave.acoustics.time_harmonic`

## `helmholtz_solver`

Solves the Helmholtz equation on linear representations, using GMRES.

```math
-\frac{\omega^2}{c_0^2}P = \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla P + \frac{2i\omega^3\alpha_0}{c_0} P - i \omega S_M.
```

{{ implementations('jwave.acoustics.time_harmonic', 'helmholtz_solver') }}
