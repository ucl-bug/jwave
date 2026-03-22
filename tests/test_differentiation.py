# Differentiation test suite for jwave
#
# Verifies jax.grad works through all solvers w.r.t. differentiable inputs.
# Tests: jit+grad finiteness, finite-difference correctness, vmap+grad.
#
# Related GitHub issues: #150, #168, #187, #189

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import born_series, helmholtz_solver
from jwave.acoustics.time_varying import (
    TimeWavePropagationSettings,
    simulate_wave_propagation,
)
from jwave.geometry import Domain, Medium, TimeAxis

FD_RTOL = 5e-2
FD_ATOL = 1e-5

HELMHOLTZ_N = (32, 32)
HELMHOLTZ_DX = (1e-3, 1e-3)
TD_N = (48, 48)
TD_DX = (1e-4, 1e-4)


def _assert_finite(x, name="gradient"):
    for leaf in jax.tree.leaves(x):
        assert jnp.all(jnp.isfinite(leaf)), f"{name} contains NaN or Inf"


def _fd_check_scalar_wrt_scalar(f, x, eps):
    grad_ad = float(jax.jit(jax.grad(f))(x))
    grad_fd = (float(f(x + eps)) - float(f(x - eps))) / (2 * eps)
    return grad_ad, grad_fd


def _assert_fd_close(ad, fd, name, rtol=FD_RTOL, atol=FD_ATOL):
    denom = max(abs(fd), atol)
    rel = abs(ad - fd) / denom
    assert rel < rtol, (
        f"{name} FD mismatch: AD={ad:.6e}, FD={fd:.6e}, rel={rel:.4f}"
    )


def _make_helmholtz_setup():
    domain = Domain(HELMHOLTZ_N, HELMHOLTZ_DX)

    c = np.ones(HELMHOLTZ_N, dtype=np.float32) * 1500.0
    c[12:20, 12:20] = 1650.0

    rho = np.ones(HELMHOLTZ_N, dtype=np.float32) * 1000.0
    rho[12:20, 12:20] = 1200.0

    alpha = np.zeros(HELMHOLTZ_N, dtype=np.float32)

    src = jnp.zeros(HELMHOLTZ_N, dtype=jnp.complex64).at[8, 8].set(1.0 + 0j)

    return dict(
        domain=domain,
        sound_speed=FourierSeries(c, domain),
        density=FourierSeries(rho, domain),
        attenuation=FourierSeries(alpha, domain),
        src=FourierSeries(src, domain),
        omega=jnp.float32(1e6),
    )


def _helmholtz_loss(sound_speed, density, attenuation, omega, src, domain,
                    checkpoint=True):
    medium = Medium(domain, sound_speed, density, attenuation, pml_size=8)
    field = helmholtz_solver(medium, omega, src, tol=1e-4,
                             restart=5, maxiter=50, checkpoint=checkpoint)
    return jnp.sum(jnp.abs(field.on_grid) ** 2).real


def _check_helmholtz_grad(param_key):
    """Run jit(grad(loss)) w.r.t. a single Helmholtz parameter."""
    s = _make_helmholtz_setup()
    keys = ["sound_speed", "density", "attenuation", "omega", "src"]
    assert param_key in keys, f"Unknown param_key: {param_key}"

    def f(x):
        args = {k: (x if k == param_key else s[k]) for k in keys}
        return _helmholtz_loss(**args, domain=s["domain"])

    g = jax.jit(jax.grad(f))(s[param_key])
    _assert_finite(g, f"helmholtz_grad_{param_key}")


class TestHelmholtzGrad:

    @pytest.mark.parametrize("param", ["sound_speed", "density", "omega", "src"])
    def test_jit_grad(self, param):
        _check_helmholtz_grad(param)

    def test_jit_grad_checkpoint_false(self):
        s = _make_helmholtz_setup()

        def f(c):
            return _helmholtz_loss(c, s["density"], s["attenuation"],
                                   s["omega"], s["src"], s["domain"],
                                   checkpoint=False)

        g = jax.jit(jax.grad(f))(s["sound_speed"])
        _assert_finite(g, "helmholtz_grad_checkpoint_false")

    def test_checkpoint_true_and_false_agree(self):
        """Regression: checkpoint should not change gradient values (#150)."""
        s = _make_helmholtz_setup()

        def make_f(ckpt):
            def f(c):
                return _helmholtz_loss(c, s["density"], s["attenuation"],
                                       s["omega"], s["src"], s["domain"],
                                       checkpoint=ckpt)
            return f

        g_true = jax.jit(jax.grad(make_f(True)))(s["sound_speed"])
        g_false = jax.jit(jax.grad(make_f(False)))(s["sound_speed"])
        diff = float(jnp.max(jnp.abs(g_true.on_grid - g_false.on_grid)))
        assert diff < 1e-5, f"checkpoint=True/False gradients differ by {diff}"


def _born_loss(sound_speed, src, omega, domain):
    medium = Medium(domain, sound_speed, pml_size=8)
    field = born_series(medium, src, omega=omega, max_iter=50, tol=1e-4)
    return jnp.sum(jnp.abs(field.on_grid) ** 2).real


class TestBornSeriesGrad:
    """born_series uses lax.while_loop which does not support reverse-mode AD.
    See GitHub issue #189 (CBS adjoint) for the planned fix.
    """

    @pytest.mark.xfail(
        reason="born_series uses lax.while_loop — no jax.grad support (#189)",
        raises=ValueError,
        strict=True,
    )
    @pytest.mark.parametrize("param", ["sound_speed", "omega"])
    def test_jit_grad(self, param):
        s = _make_helmholtz_setup()
        if param == "sound_speed":
            def f(c):
                return _born_loss(c, s["src"], s["omega"], s["domain"])
            g = jax.jit(jax.grad(f))(s["sound_speed"])
        else:
            def f(w):
                return _born_loss(s["sound_speed"], s["src"], w, s["domain"])
            g = jax.jit(jax.grad(f))(s["omega"])
        _assert_finite(g, f"born_grad_{param}")


def _make_td_setup(field_type="fourier"):
    from jwave import OnGrid
    from jwave.geometry import circ_mask

    if field_type == "fourier":
        FieldCls = FourierSeries
    elif field_type == "ongrid":
        FieldCls = OnGrid
    else:
        raise ValueError(f"Unknown field_type: {field_type!r}")

    domain = Domain(TD_N, TD_DX)

    c = np.ones(TD_N, dtype=np.float32) * 1500.0
    c[16:32, 16:32] = 1600.0

    rho = np.ones(TD_N, dtype=np.float32) * 1000.0
    rho[16:32, 16:32] = 1100.0

    sound_speed = FieldCls(c, domain)
    density = FieldCls(rho, domain)

    medium = Medium(domain, sound_speed, density, pml_size=8)
    time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=2e-6)

    p0_np = 5.0 * circ_mask(TD_N, 3, (24, 24))
    p0_np = np.expand_dims(p0_np, -1).astype(np.float32)

    return dict(
        domain=domain, sound_speed=sound_speed, density=density,
        medium=medium, time_axis=time_axis, p0=FieldCls(p0_np, domain),
        settings=TimeWavePropagationSettings(checkpoint=False),
    )


def _td_loss(sound_speed, density, p0, domain, time_axis, settings):
    medium = Medium(domain, sound_speed, density, pml_size=8)
    fields = simulate_wave_propagation(medium, time_axis, p0=p0,
                                       settings=settings)
    return jnp.mean(jnp.abs(fields[-1].on_grid) ** 2)


def _check_td_grad(field_type, param_key):
    """Run jit(grad(loss)) w.r.t. a single TD parameter."""
    s = _make_td_setup(field_type)

    def f(x):
        args = dict(sound_speed=s["sound_speed"], density=s["density"],
                    p0=s["p0"], domain=s["domain"],
                    time_axis=s["time_axis"], settings=s["settings"])
        args[param_key] = x
        return _td_loss(**args)

    g = jax.jit(jax.grad(f))(s[param_key])
    _assert_finite(g, f"td_{field_type}_grad_{param_key}")


class TestTimeDomainFourierGrad:

    @pytest.mark.parametrize("param", ["sound_speed", "density", "p0"])
    def test_jit_grad(self, param):
        _check_td_grad("fourier", param)

    def test_jit_grad_with_checkpoint(self):
        s = _make_td_setup("fourier")
        settings_ckpt = TimeWavePropagationSettings(checkpoint=True)

        def f(c):
            return _td_loss(c, s["density"], s["p0"], s["domain"],
                            s["time_axis"], settings_ckpt)

        g = jax.jit(jax.grad(f))(s["sound_speed"])
        _assert_finite(g, "td_fourier_grad_checkpoint")


class TestTimeDomainOnGridGrad:
    """OnGrid solver uses diag_jacobian which has no OnGrid dispatch.
    All tests xfail until dispatch is added.
    """

    @pytest.mark.xfail(
        reason="diag_jacobian has no OnGrid dispatch",
        strict=True,
    )
    @pytest.mark.parametrize("param", ["sound_speed", "density", "p0"])
    def test_jit_grad(self, param):
        _check_td_grad("ongrid", param)


class TestFiniteDifferenceChecks:

    def test_helmholtz_fd_omega(self):
        s = _make_helmholtz_setup()

        def f(w):
            return _helmholtz_loss(s["sound_speed"], s["density"],
                                   s["attenuation"], w, s["src"], s["domain"])

        grad_ad, grad_fd = _fd_check_scalar_wrt_scalar(f, s["omega"], eps=1e2)
        _assert_fd_close(grad_ad, grad_fd, "helmholtz_omega")

    def test_td_fourier_fd_sound_speed_pixels(self):
        from jwave.geometry import circ_mask

        N = (24, 24)
        dx = (1e-4, 1e-4)
        domain = Domain(N, dx)
        c_np = np.ones(N, dtype=np.float32) * 1500.0
        c_np[10:14, 10:14] = 1600.0
        p0_np = np.expand_dims(
            5.0 * circ_mask(N, 2, (12, 12)), -1
        ).astype(np.float32)
        rho = FourierSeries(np.ones(N, dtype=np.float32) * 1000.0, domain)
        p0 = FourierSeries(p0_np, domain)
        medium_kwargs = dict(pml_size=6)
        time_axis = TimeAxis(dt=1.5e-8, t_end=1e-6)
        settings = TimeWavePropagationSettings(checkpoint=False)

        def loss(c_flat):
            c = FourierSeries(c_flat.reshape(N), domain)
            medium = Medium(domain, c, rho, **medium_kwargs)
            fields = simulate_wave_propagation(medium, time_axis, p0=p0,
                                               settings=settings)
            return jnp.mean(jnp.abs(fields[-1].on_grid) ** 2)

        c_flat = jnp.array(c_np.ravel())
        grad_ad = np.asarray(jax.jit(jax.grad(loss))(c_flat))

        eps = 1.0
        for idx in [0, N[0] * N[1] // 2, N[0] * 11 + 12]:
            c_p = c_flat.at[idx].set(c_flat[idx] + eps)
            c_m = c_flat.at[idx].set(c_flat[idx] - eps)
            fd = (float(loss(c_p)) - float(loss(c_m))) / (2 * eps)
            ad = float(grad_ad[idx])
            if abs(fd) > FD_ATOL:
                _assert_fd_close(ad, fd, f"td_fourier_c_pixel_{idx}")


class TestVmapGrad:

    def test_helmholtz_vmap_grad_omega(self):
        s = _make_helmholtz_setup()

        def loss_single(omega):
            return _helmholtz_loss(s["sound_speed"], s["density"],
                                   s["attenuation"], omega, s["src"],
                                   s["domain"])

        omegas = jnp.array([0.9e6, 1.0e6, 1.1e6], dtype=jnp.float32)
        grads = jax.jit(jax.vmap(jax.grad(loss_single)))(omegas)
        _assert_finite(grads, "vmap_grad_omega")
        assert grads.shape == (3,)

    def test_td_fourier_vmap_grad_p0_amplitude(self):
        s = _make_td_setup("fourier")

        def loss_single(amp):
            return _td_loss(s["sound_speed"], s["density"], s["p0"] * amp,
                            s["domain"], s["time_axis"], s["settings"])

        amps = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
        grads = jax.jit(jax.vmap(jax.grad(loss_single)))(amps)
        _assert_finite(grads, "vmap_grad_p0_amp")
        assert grads.shape == (3,)
