# Solving Helmholtz problems

> The models used by the Helmholtz solvers are derived with [Santeri Kaupinmäki](https://bug.medphys.ucl.ac.uk/santeri-kaupinmaki)

Depending on the heterogeneity of the model, the Helmholtz solver automatically calls different functions to calculate the Helmholtz operator. The solution is always found using an iterative linear solver with matrix-free linear opeerators.

## The Homogeneous density case
For a medium with homogeneous density and attenuation, the wave equations is given as 

```math
\left( \nabla^2 - \frac{1}{c_0^2}\frac{\partial^2}{\partial t^2}\right)p = -\frac{\partial}{\partial t}S_M
```

Taking the Fourier transform in time, and denoting the Fourier transform of a function $`f`$ as $`\hat f`$, one gets the Helmholtz equation

```math
\left( \nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p = -i\omega \hat S_M.
```

Note that there both $`\hat p`$ and $`\hat S_M`$ are complex.

### Perfectly matched layer
To enforce the Sommerfield radiation condition, the components of the gradient operator $`\hat \nabla`$ are modified as[^1]

```math
\frac{\partial}{\partial x_j} \to \frac{1}{\gamma_j} \frac{\partial}{\partial x_j},
\qquad
\gamma_j = 1 + \frac{1}{k_0}\sigma_j(x), \qquad \sigma_j(x)\frac{1}{\|x_j - x_j^{PML}\|}
```

and the equations solved is 

```math
\left( \hat\nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p = -i\omega \hat S_M.
```

## Heterogeneous density
For heterogeneous densities, we have to start from the conservation equations

```math
\begin{aligned}
\frac{\partial u}{\partial t} &= -\frac{1}{\rho_0}\nabla p \\
\frac{\partial \rho}{\partial t} &= -\rho_0 \nabla \cdot u - u \cdot \nabla \rho_0 + S_M\\
p &= c_0^2(\rho + d\cdot \nabla \rho_0)
\end{aligned}
```

Taking them together gives

```math
\left[\rho_0\nabla\cdot \left( \frac{1}{\rho_0}\nabla \right)-\frac{1}{c_0^2}\frac{\partial^2}{\partial t^2}\right]\hat p = - \frac{\partial S_M}{\partial t}
```

The Fourier transform of this expression gives the Helmholtz equation with heterogeneous density
```math
\left[\rho_0\nabla\cdot \left( \frac{1}{\rho_0}\nabla \right)+\frac{\omega^2}{c_0^2}\right]\hat p = -i\omega \hat S_M.
```

## Attenuation
If attenuation is modeled using Stokes absorption, the Helmholtz equation is given by 

```math
\left[(1 + i\tau \omega)\nabla^2 + \frac{\omega^2}{c_0^2}\right]\hat p =  \left(\omega^2\tau- i\omega\right)S_M
```

## General case
In the general case of both heterogeneous attenuation and density, the system of first order equations is

```math
\begin{aligned}
\frac{\partial u}{\partial t} &= -\frac{1}{\rho_0}\nabla p \\
\frac{\partial \rho}{\partial t} &= -\rho_0 \nabla \cdot u - u \cdot \nabla \rho_0 + S_M\\
p &= c_0^2(\rho + d\cdot \nabla \rho_0 + \tau\frac{\partial}{\partial t} \rho)
\end{aligned}
```

Taking the second order time derivative of the last row gives

```math
\frac{\partial^2 p}{\partial t^2} = c_0^2\left(\frac{\partial^2 \rho}{\partial t^2} + \frac{\partial u}{\partial t}\cdot \nabla \rho_0 + \tau \frac{\partial }{\partial t} \frac{\partial^2 \rho}{\partial t^2} \right) 
```

Plugging the frist equation gives


```math
\frac{\partial^2 p}{\partial t^2} = c_0^2\left[\frac{\partial^2 \rho}{\partial t^2} - \left(\frac{1}{\rho_0} \nabla p \right)\cdot \nabla \rho_0 + \tau \frac{\partial }{\partial t} \frac{\partial^2 \rho}{\partial t^2} \right] 
```

By taking the gradient of the mass conservation equation and plugging in the momentum conservation equation gives


```math
\begin{aligned}
\frac{\partial^2 \rho}{\partial t^2}  &= -\rho_0 \nabla \cdot \frac{\partial u}{\partial t}  -  \frac{\partial u}{\partial t} \cdot \nabla \rho_0 - \frac{\partial S_M}{\partial t} \\
&= \rho_0 \nabla \cdot \left(\frac{1}{\rho_0}\nabla p\right) +  \left(\frac{1}{\rho_0}\nabla p\right) \cdot \nabla \rho_0 - \frac{\partial S_M}{\partial t} \\
&= \rho_0 \nabla \cdot \left(\frac{1}{\rho_0}\nabla p\right) +   \left( \nabla \rho_0\right) \cdot\left(\frac{1}{\rho_0}\nabla p\right) - \frac{\partial S_M}{\partial t}
\end{aligned}
```

where in the last equation is using the commutative property of the dot product on real Hilbert spaces.

This last equation can be used to remove all $`\partial^2_{t}\rho`$ terms, obtaining the wave equation for pressure

```math
\begin{aligned}
\frac{\partial^2 p}{\partial t^2} &= c_0^2\left[\rho_0 \nabla \cdot \left(\frac{1}{\rho_0}\nabla p\right) - \frac{\partial S_M}{\partial t} + \tau \frac{\partial }{\partial t} \frac{\partial^2 \rho}{\partial t^2} \right] \\
&= c_0^2\left\{\rho_0 \nabla \cdot \left(\frac{1}{\rho_0}\nabla p\right) - \frac{\partial S_M}{\partial t} + \tau \frac{\partial }{\partial t} \left[ \Big(\rho_0 \nabla + \nabla \rho_0\Big) \cdot\left(\frac{1}{\rho_0}\nabla p\right) - \frac{\partial S_M}{\partial t} \right] \right\}
\end{aligned}
```

Taking the Fourier transform and rearranging terms gives the Helmholtz equation for fully heterogeneous media

```math
\left\{\left[\rho_0\nabla + i\omega\tau\Big(\rho_0 \nabla  +  \nabla \rho_0\Big)  \right] \cdot\left(\frac{1}{\rho_0}\nabla \right) + \frac{\omega^2}{c_0^2} \right\}p = - i\omega( 1 + i\omega\tau) S_M.
```

Which can be written in the more familiar form 

```math
\left(\hat \nabla_{\omega}^2 + \frac{\omega^2}{c_0^2} \right)p = \left(\omega^2\tau- i\omega\right)S_M
```
by defining

```math
\hat \nabla_{\omega}^2 = \bigg[\rho_0\nabla + i\omega\tau\Big(\rho_0 \nabla  +  \nabla \rho_0\Big)  \bigg] \cdot\left(\frac{1}{\rho_0}\nabla \right)
```

## References

[^1]: [http://oomph-lib.maths.man.ac.uk/doc/pml_helmholtz/scattering/latex/refman.pdf](http://oomph-lib.maths.man.ac.uk/doc/pml_helmholtz/scattering/latex/refman.pdf)