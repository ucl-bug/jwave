# Changelog


## (latest)

### Bug Fix

* Correctly handles Nyquist frequency for Helmholtz operator, to improve agreement with k-Wave. [Antonio Stanziola]

* Incorrect domain size for angular spectrum. [Antonio Stanziola]

* Angular spectrum is only dispatched on 'pressure' type. [Antonio Stanziola]

### Features

* Convergent born series. [Antonio Stanziola]


## 0.0.3 (2022-07-05)

### Features

* Angular spectrum method for single frequency sources. [Antonio Stanziola]

* Differentiable rayleigh integral (from a plane) [Antonio Stanziola]


## 0.0.2 (2022-06-23)

### Bug Fix

* Helmholtz notebook parameters bug. [Antonio Stanziola]

### Features

* Generate TimeHarmonicSource from point sources. [Antonio Stanziola]


## 0.0.1 (2022-06-07)

### Bug Fix

* Using numpy operations in TimeAxis for static fields. [Antonio Stanziola]

* Pml for 1D and 3D simulations. [Antonio Stanziola]

* Plotting functions of jwave.utils now work with both Fields and arrays. [Antonio Stanziola]

### Features

* Finite differences helmholtz tested. [Antonio Stanziola]

* Extract time varying params without running the simulation. [Antonio Stanziola]

* Windows one-line installer. [Antonio Stanziola]

### Tests

* Removed flake8 testing for E111. [Antonio Stanziola]
