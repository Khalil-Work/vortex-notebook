"""Verification tests for spectral_flow.

Each test pins a distinct property of the method:

- exact spectral recovery of the Taylor-Green velocity from its vorticity
  (streamfunction operator, zero mode);
- machine-precision divergence freedom of the recovered velocity;
- accuracy of the Taylor-Green decay (viscous term + RK4 + operators);
- fourth-order time-convergence slope;
- conservation of mean vorticity under the nonlinear term;
- agreement of the de-aliased spectral nonlinear operator with an
  independent finite-difference evaluation on a fixed field;
- exact analytic energy/enstrophy decay rates;
- structure of the 2/3-rule dealiasing mask.
"""

import numpy as np
import pytest

from spectral_flow import SpectralVorticitySolver, taylor_green_vorticity
from spectral_flow.solver import taylor_green_velocity


def grid(n):
    x = np.arange(n) * (2.0 * np.pi / n)
    return np.meshgrid(x, x, indexing="ij")


@pytest.fixture(scope="module")
def tg_solver():
    return SpectralVorticitySolver(n=64, viscosity=0.01)


def test_velocity_recovery_from_taylor_green_vorticity(tg_solver):
    xx, yy = grid(tg_solver.n)
    w0 = taylor_green_vorticity(xx, yy)
    w_hat = np.fft.fft2(w0)
    u, v = tg_solver.velocity(w_hat)
    ue, ve = taylor_green_velocity(xx, yy, 0.0, tg_solver.nu)
    np.testing.assert_allclose(u, ue, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v, ve, rtol=1e-12, atol=1e-12)


def test_velocity_is_divergence_free_to_machine_precision(tg_solver):
    # Spectral divergence i(kx u_hat + ky v_hat) must vanish identically.
    xx, yy = grid(tg_solver.n)
    rng = np.random.default_rng(7)
    w = np.cos(xx) * np.cos(2 * yy) + 0.3 * rng.standard_normal((tg_solver.n,) * 2)
    w_hat = np.fft.fft2(w)
    u_hat, v_hat = tg_solver.velocity_hat(w_hat)
    div_hat = 1j * (tg_solver.kx * u_hat + tg_solver.ky * v_hat)
    assert np.max(np.abs(div_hat)) < 1e-10 * np.max(np.abs(w_hat))


def test_taylor_green_decay_accuracy(tg_solver):
    dt, t_end = 0.002, 1.0
    xx, yy = grid(tg_solver.n)
    w0 = taylor_green_vorticity(xx, yy)
    history = tg_solver.integrate(w0, dt=dt, nsteps=int(t_end / dt))
    t, w_hat = history[-1]
    assert t == pytest.approx(t_end)
    assert tg_solver.velocity_l2_error(w_hat, t_end) < 1e-7


def test_fourth_order_time_convergence():
    # Parameters chosen so RK4 truncation stands well above the ~1e-15
    # spectral roundoff floor while remaining inside the stability range of
    # the excited Taylor-Green eigenvalue (decay rate 2*nu = 1/s, so
    # |lambda|*dt <= 0.2). Fine-step saturation is not reached at dt=0.025,
    # where the error is ~1e-9.
    solver = SpectralVorticitySolver(n=8, viscosity=0.5)
    xx, yy = grid(solver.n)
    w0 = taylor_green_vorticity(xx, yy)
    errors = []
    for dt in (0.2, 0.1, 0.05, 0.025):
        nsteps = int(round(1.0 / dt))
        _, w_hat = solver.integrate(w0, dt=dt, nsteps=nsteps)[-1]
        errors.append(solver.velocity_l2_error(w_hat, nsteps * dt))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 3.5), f"observed RK4 orders {orders}"


def test_mean_vorticity_conserved_under_nonlinear_dynamics():
    solver = SpectralVorticitySolver(n=64, viscosity=1e-3)
    xx, yy = grid(solver.n)
    w0 = np.cos(xx) * np.cos(yy) + 0.7 * np.sin(2 * xx) * np.cos(yy)
    history = solver.integrate(w0, dt=0.002, nsteps=200, sample_every=100)
    for _, w_hat in history:
        assert abs(w_hat[0, 0]) < 1e-10 * max(1.0, np.max(np.abs(w_hat)))


def test_nonlinear_operator_matches_finite_differences():
    solver = SpectralVorticitySolver(n=64, viscosity=0.01, dealias=True)
    xx, yy = grid(solver.n)
    # Smooth band-limited field: the 2/3 truncation leaves it untouched, and
    # a second-order central finite-difference reference is accurate enough
    # to catch form/sign errors in the spectral operator.
    w = np.cos(xx) * np.cos(yy) + 0.5 * np.sin(2 * xx) * np.sin(yy)
    w_hat = np.fft.fft2(w)
    u, v = solver.velocity(w_hat)
    uw, vw = u * w, v * w
    dx = 2.0 * np.pi / solver.n
    div_fd = (np.roll(uw, -1, axis=0) - np.roll(uw, 1, axis=0)) / (2 * dx) + (
        np.roll(vw, -1, axis=1) - np.roll(vw, 1, axis=1)
    ) / (2 * dx)
    rhs = np.fft.ifft2(solver.nonlinear_hat(w_hat)).real
    scale = np.max(np.abs(div_fd))
    # The central-difference reference itself carries O(dx^2) truncation
    # (~1.5% at this grid for the mode-2 content), so the comparison
    # tolerance bounds FD truncation, not spectral accuracy.
    assert np.max(np.abs(rhs + div_fd)) / scale < 3e-2


def test_energy_and_enstrophy_decay_exactly_for_taylor_green(tg_solver):
    nu, t_end, dt = tg_solver.nu, 1.0, 0.002
    xx, yy = grid(tg_solver.n)
    w0 = taylor_green_vorticity(xx, yy)
    _, w_hat = tg_solver.integrate(w0, dt=dt, nsteps=int(t_end / dt))[-1]
    decay = np.exp(-4.0 * nu * t_end)          # both E and Z scale with f^2
    assert tg_solver.energy(w_hat) == pytest.approx(0.25 * decay, rel=1e-8)
    assert tg_solver.enstrophy(w_hat) == pytest.approx(0.5 * decay, rel=1e-8)


def test_dealias_mask_structure():
    # Retained band is |k| < N/3 STRICTLY per direction: on N=48 the
    # boundary mode 16 is excluded (inclusive cutoff would fold 16+16=32
    # onto the retained alias -16) and mode 15 is retained.
    solver = SpectralVorticitySolver(n=48, viscosity=0.01)
    assert not solver.dealias_mask[16, 0]
    assert solver.dealias_mask[15, 0]
    assert solver.dealias_mask[1, 1]
    kept = solver.dealias_mask.sum()
    assert 0 < kept < solver.k2.size


def test_curl_round_trip_recovers_vorticity_at_all_retained_modes():
    # omega -> psi -> (u, v) -> spectral curl must return omega exactly on
    # every retained mode (including the high ones) for a zero-mean
    # band-limited field.
    solver = SpectralVorticitySolver(n=64, viscosity=0.01)
    rng = np.random.default_rng(11)
    w_hat = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)))
    w_hat *= solver.dealias_mask
    w_hat[0, 0] = 0.0
    u_hat, v_hat = solver.velocity_hat(w_hat)
    curl_hat = 1j * (solver.kx * v_hat - solver.ky * u_hat)
    np.testing.assert_allclose(
        curl_hat, w_hat, rtol=1e-12, atol=1e-9 * np.max(np.abs(w_hat))
    )


def test_nonlinear_operator_matches_padded_reference():
    # Behavioral alias check: evaluate -div(u*omega) exactly with the 3/2-rule
    # (zero-padded 72-point grid for N=48), restrict to the coarse grid,
    # apply the same retained-band truncation, and compare. Random retained
    # band-limited input exercises genuine mode interactions, including the
    # high-high products that would alias onto retained modes under a
    # non-strict cutoff.
    n = 48
    solver = SpectralVorticitySolver(n=n, viscosity=0.01, dealias=True)
    rng = np.random.default_rng(23)
    w_hat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    w_hat *= solver.dealias_mask

    m = 3 * n // 2  # 72; products of retained modes reach |k| = 30 < m/2
    k = np.fft.fftfreq(m, d=1.0 / m)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    inv = np.zeros_like(k2)
    inv[k2 > 0] = 1.0 / k2[k2 > 0]

    W = np.zeros((m, m), dtype=complex)
    half = n // 2
    W[:half, :half] = w_hat[:half, :half]
    W[:half, -half:] = w_hat[:half, -half:]
    W[-half:, :half] = w_hat[-half:, :half]
    W[-half:, -half:] = w_hat[-half:, -half:]

    psi = inv * W
    u_hat = 1j * ky * psi
    v_hat = -1j * kx * psi
    w = np.fft.ifft2(W).real
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real
    exact = -(1j * kx * np.fft.fft2(u * w) + 1j * ky * np.fft.fft2(v * w))

    reference = np.zeros((n, n), dtype=complex)
    reference[:half, :half] = exact[:half, :half]
    reference[:half, -half:] = exact[:half, -half:]
    reference[-half:, :half] = exact[-half:, :half]
    reference[-half:, -half:] = exact[-half:, -half:]
    # The padded-grid fft2/ifft2 pair carries its own m^2 normalization: with
    # coefficients embedded unscaled, extracted spectra are smaller by
    # (n/m)^2 = 4/9, so rescale to coarse-grid coefficient convention.
    reference *= (m / n) ** 2
    reference *= solver.dealias_mask

    got = solver.nonlinear_hat(w_hat)
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(got - reference)) / scale < 1e-12


def test_mean_vorticity_is_passive_and_velocity_curl_has_zero_mean():
    # omega_hat(0,0) is exactly conserved; the recovered velocity cannot
    # represent a uniform background curl, so its curl is the zero-mean part.
    solver = SpectralVorticitySolver(n=32, viscosity=1e-2)
    xx, yy = grid(32)
    w0 = 2.0 + np.cos(xx) * np.cos(yy)  # nonzero mean vorticity
    history = solver.integrate(w0, dt=0.002, nsteps=100, sample_every=50)
    w0_hat = np.fft.fft2(w0) * solver.dealias_mask
    for _, w_hat in history:
        assert w_hat[0, 0] == pytest.approx(w0_hat[0, 0], rel=1e-12)
    u_hat, v_hat = solver.velocity_hat(history[-1][1])
    assert abs(1j * (solver.kx * v_hat - solver.ky * u_hat)[0, 0]) < 1e-10


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        SpectralVorticitySolver(n=63, viscosity=0.01)
    with pytest.raises(ValueError):
        SpectralVorticitySolver(n=64, viscosity=0.0)
