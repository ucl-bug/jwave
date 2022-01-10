



def waveholtz_iteration(
    medium: geometry.Medium,
    omega: float,
    discretization: FourierSeries,
    source: Union[None, jnp.ndarray] = None,
    cfl: float = 0.1,
):
    """
    Constructs a wave solver that runs for one period, and
    integrates the results according to eq. (2.4) of
    https://arxiv.org/pdf/1910.10148.pdf
    """

