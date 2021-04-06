from jax.config import config


def detect_nans(show_warning=True):
    print(
        "X" * 30
        + "\nWARNING: NaNs detection is enabled, this likely causes serious performance losses\n"
        + "X" * 30
    )
    config.update("jax_debug_nans", True)
