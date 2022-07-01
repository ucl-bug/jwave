# `jwave.acoustics.time_harmonic`

## `helmholtz_solver`

Solves the Helmholtz equation on linear representations, using GMRES.

```math
-\frac{\omega^2}{c_0^2}P = \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla P + \frac{2i\omega^3\alpha_0}{c_0} P - i \omega S_M.
```

{{ implementations('jwave.acoustics.time_harmonic', 'helmholtz_solver') }}

## `rayleigh_integral`

Rayleigh integral for a given pressure field on a finite plane. See eq. (A.2) in [[Sapozhnikov et al.](https://asa.scitation.org/doi/pdf/10.1121/1.4928396)].

```math
u(\mathbf{r}) = \frac{1}{2\pi}\int_\Sigma u(\mathbf{x})
\partial_n \left( \frac{e^{ik_0|\mathbf{r} - \mathbf{x}|}}{|\mathbf{r} - \mathbf{x}|} \right) d\Sigma(\mathbf{x})
```

where $`\Sigma`$ is the domain of integration, corresponding to the finite plane where $`u`$ is defined (the algorithm assumes that the field is identically zero outside such domain), and $`k_0`$ is the wavenumber of the wave.

{{ implementations('jwave.acoustics.time_harmonic', 'rayleigh_integral') }}
