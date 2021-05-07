Implements various operators in the Fourier domain.

A generic spectral operator is given by a function that takes two inputs:

1. An input array representing the signal in the un-transformed domain (e.g. spatial domain) on which the operator acts
2. A `geometry.kGrid` object containing various arrays that support spectral calculations, such as the `k-space` frequency axis

The reason for esplicitly asking the `kGrid` object is memory optimization. While it would be possible to compile the spectra functions such that the `kGrid` doesn't have to be explicitly passed, because many different spectral operations rely on the same grid object (.e.g. the laplacian and $`\partial_x`$ operator both need the $`k_x`$ axis) it is convenient for them to refer to a common object.

Many functions in this module are constructors of new functions, or meta-functions. They are used to construct a specific spectral operator, while at the same time making sure that the `kGrid` object is populated with the required information.

!!! example

    To take the derivative of an image, the step is to construct the grid object
    
    ```python
    from jwave.geometry import kGrid
    N = img.shape
    dx = tuple([1 for _ in N])
    grid = kGrid.make_grid(N,dx)
    ```

    We then initialize the derivative operator and update the grid structure

    ```python
    from jwave import spectral
    Dx, grid = spectral.derivative_init(grid, sample_input=img, axis=0)
    ```

    And then apply the operator to our image

    ```python
    Dx_img = Dx(img, grid)
    ```

    If we define a new operator, the updated `grid` can still be used as input
    to the first one

    ```python
    from jwave.geometry import Staggered
    Dx_staggered , grid = derivative_init(grid, img, axis=0, staggered=Staggered.FORWARD)

    Dx_stag_img = Dx_staggered(img, grid)
    Dx_img = Dx(img, grid)  # This still works
    ```

    More examples are given in the [Taking derivatives notebook](../examples/spectral_operators/)

::: jwave.spectral
    handler: python
    members:
        - derivative_init
    show_root_heading: true
    show_source: false