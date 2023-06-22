# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

[Unreleased]: https://github.com/ucl-bug/jwave/compare/0.1.0...master
[0.1.0]: https://github.com/ucl-bug/jwave/compare/0.0.4...0.1.0
[0.0.4]: https://github.com/ucl-bug/jwave/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/ucl-bug/jwave/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/ucl-bug/jwave/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/ucl-bug/jwave/releases/tag/0.0.1
