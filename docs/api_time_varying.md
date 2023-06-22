# `jwave.acoustics.time_varying`

## `mass_conservation_rhs`

Implements the mass conservation equation with split fields, see eq. (2.1), (2.8) and (2.17) of the [k-Wave manual](http://www.k-wave.org/manual/k-wave_user_manual_1.1.pdf), which is given by

$$
\frac{\partial}{\partial t} \mathbf{u_\varepsilon} = - \rho_0 \frac{\partial}{\partial \varepsilon} u_\varepsilon + S_M
$$

where $u_\varepsilon$ is the velocity field for the $\varepsilon$ coordinate, $\rho_0$ is the ambient density, and $S_M$ is the mass source term

{{ implementations('jwave.acoustics.time_varying', 'mass_conservation_rhs') }}


## `momentum_conservation_rhs`

Implements the momentum conservation equation, see eq. (2.1) of the [k-Wave manual](http://www.k-wave.org/manual/k-wave_user_manual_1.1.pdf), which is given by

$$
\frac{\partial}{\partial t} \mathbf{u} = - \frac{1}{\rho_0} \nabla p
$$

where $\rho_0$ is the background density, $p$ is the pressure, and $\mathbf{u}$ is the velocity field. Note that this operator expects both the pressure field and velocity field as input, to ease the implementation of other customized variants that may depend on both fields.

{{ implementations('jwave.acoustics.time_varying', 'momentum_conservation_rhs') }}

---

## `pressure_from_density`

Evaluates the pressure field from the acoustic density, as

$$
p = c_0^2 \sum_{\varepsilon} \rho_\varepsilon
$$

where $c_0$ is the speed of sound, $\rho_\varepsilon$ is the acoustic density, and $\varepsilon$ is the coordinate.

{{ implementations('jwave.acoustics.time_varying', 'pressure_from_density') }}

---

## `simulate_wave_propagation`

Simulates wave propagation by integrating the system of equations described in eq. (2.1) of the [k-Wave manual](http://www.k-wave.org/manual/k-wave_user_manual_1.1.pdf), using a first-order symplectic integrator.

{{ implementations('jwave.acoustics.time_varying', 'simulate_wave_propagation') }}
