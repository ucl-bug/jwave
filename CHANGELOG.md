# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2023-06-29
### Changed
- Refactored `save_video` to use opencv.

### Deprecated
- `plot_complex_field` has been deprecated in favor of `display_complex_field`

### Removed
- Removed the uncertainty propagation notebook example. For a more in depth example of using linear uncertainty propagation see [this repository](https://github.com/ucl-bug/linear-uncertainty)

### Added
- Exposed `points_on_circle` function to generate points on a circle
- Exposed `unit_fibonacci_sphere` function
- Exposed `fibonacci_sphere` function
- Exposed `sphere_mask` function for creating spherical binary masks
- Exposed `circ_mask` function for creating circular binary masks
- Exposed bli_function that is used to compute the band limited interpolant

## [0.1.3] - 2023-06-28
### Added
- Added off grid sensors [@tomelse]

## [0.1.2] - 2023-06-22
### Changed
- updated documentation
- made imageio and tqdm optional dependencies

## [0.1.1] - 2023-06-22
### Fixed
- fixed pypi classifiers

## [0.1.0] - 2023-06-22
### Added
- `k0` is automatically calculated in the Convergent Born Series, if not given, using the fromula from Osnabrugge et al.

### Fixed
- updated for new `Array` type in `jax` 0.4.x

### Changed
- reverted checkpoint to only step checkpoints for time varying simulations. Soon jwave will use diffrax for advanced checkpointing

## [0.0.4] - 2022-11-04
### Added
- Convergent Born series.

### Fixed
- Correctly handles Nyquist frequency for Helmholtz operator, to improve agreement with k-Wave.
- Fixed incorrect domain size for angular spectrum.
- Angular spectrum is only dispatched on `pressure` types.

## [0.0.3] - 2022-07-05
### Added
- Angular spectrum method for single frequency sources.
- Differentiable rayleigh integral (from a plane)

## [0.0.2] - 2022-06-23
### Added
- Generate `TimeHarmonicSource` from point sources.

### Fixed
- Helmholtz notebook parameters bug.

## [0.0.1] - 2022-06-07
### Added
- Finite differences helmholtz tested.
- Extract time varying params without running the simulation.
- Windows one-line installer

### Fixed
- Using numpy operations in TimeAxis for static fields.
- Pml for 1D and 3D simulations.
- Plotting functions of `jwave.utils` now work with both `Field`s and arrays.

[Unreleased]: https://github.com/ucl-bug/jwave/compare/0.1.4...master
[0.1.4]: https://github.com/ucl-bug/jwave/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/ucl-bug/jwave/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/ucl-bug/jwave/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/ucl-bug/jwave/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/ucl-bug/jwave/compare/0.0.4...0.1.0
[0.0.4]: https://github.com/ucl-bug/jwave/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/ucl-bug/jwave/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/ucl-bug/jwave/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/ucl-bug/jwave/releases/tag/0.0.1

