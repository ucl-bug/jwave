# `jwave.acoustics.operators`


## `helmholtz`

Implements the Helmholtz operator, constructed from the second-order wave equation including Stokes absorption.

The Helmholtz operator acting on $u$ is defined as

$$
\left(\hat \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla + \frac{2i\omega^3\alpha_0}{c_0} + \frac{\omega^2}{c_0^2}\right) u
$$

where $u$ is the pressure, $\rho_0$ is the background density, $\alpha_0$ is the background absorption, $\omega$ is the angular frequency, and $c_0$ is the speed of light.

{{ implementations('jwave.acoustics.operators', 'helmholtz') }}

---

## `laplacian_with_pml`

Given a field $u$, a medium object `medium` and an angular frequency $\omega$, this operator computes the Laplacian of $u$ modified with PML defined as in [[Bermudez et al., 2007]](https://www.sciencedirect.com/science/article/pii/S0021999106004487), where each partial derivative is modified as


$$
\frac{\partial^2}{\partial \varepsilon^2} \to \frac{1}{\gamma_\varepsilon}\frac{\partial}{\partial \varepsilon}\frac{1}{\gamma_\varepsilon}\frac{\partial}{\partial \varepsilon}
$$

The operator is further augmented by a second term, which takes into account a potential heterogeneous ambient density $\rho_0$, giving the final form of

$$
\left(\hat \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla\right) u
$$

where $u$ is the pressure and $\hat \nabla$ is the modified Laplacian.

{{ implementations('jwave.acoustics.operators', 'laplacian_with_pml') }}

---

## `wavevector`

The `wavevector` operator acting on a field $u$ is defined as

$$
\left(\frac{2i\omega^3\alpha_0}{c_0} + \frac{\omega^2}{c_0^2}\right) u
$$

where $\alpha_0$ is the absorption coefficient of the medium, $c_0$ is the speed of sound in the medium, and $\omega$ is the angular frequency. The absorption coefficient is defined in units of `dB/(MHz^y cm)`.

{{ implementations('jwave.acoustics.operators', 'wavevector') }}
