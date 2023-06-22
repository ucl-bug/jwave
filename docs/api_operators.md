# `jwave.acoustics.operators`


## `helmholtz`

This implementation is for the Helmholtz operator, constructed from the second-order wave equation inclusive of Stokes absorption.

The Helmholtz operator acting on $u$ is defined as:

$$
\left(\hat \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla + \frac{2i\omega^3\alpha_0}{c_0} + \frac{\omega^2}{c_0^2}\right) u
$$

In the equation above, $u$ denotes the pressure, $\rho_0$ stands for the background density, $\alpha_0$ signifies the background absorption, $\omega$ represents the angular frequency, and $c_0$ is the speed of sound.

{{ implementations('jwave.acoustics.operators', 'helmholtz') }}

---

## `laplacian_with_pml`

Given a field $u$, a medium object `medium`, and an angular frequency $\omega$, this operator calculates the Laplacian of $u$, modified with Perfectly Matched Layer (PML) as defined in [[Bermudez et al., 2007]](https://www.sciencedirect.com/science/article/pii/S0021999106004487). Each partial derivative is altered as follows:


$$
\frac{\partial^2}{\partial \varepsilon^2} \to \frac{1}{\gamma_\varepsilon}\frac{\partial}{\partial \varepsilon}\frac{1}{\gamma_\varepsilon}\frac{\partial}{\partial \varepsilon}
$$

The operator is further enhanced by a second term, which accounts for a potential heterogeneous ambient density $\rho_0$. This results in the final form:

$$
\left(\hat \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla\right) u
$$

In the equation above, $u$ represents the pressure and $\hat \nabla$ denotes the modified Laplacian.

{{ implementations('jwave.acoustics.operators', 'laplacian_with_pml') }}

---

## `wavevector`

The `wavevector` operator, acting on a field $u$, is defined as:

$$
\left(\frac{2i\omega^3\alpha_0}{c_0} + \frac{\omega^2}{c_0^2}\right) u
$$

In this formula, $\alpha_0$ represents the medium's absorption coefficient, $c_0$ denotes the speed of sound in the medium, and $\omega$ signifies the angular frequency. The absorption coefficient is measured in units of `dB/(MHz^y cm)`.

{{ implementations('jwave.acoustics.operators', 'wavevector') }}
