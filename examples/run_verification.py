"""Produce all verification results, figures, animation, and raw data.

Runs (from the repository root, with the dependencies installed):

    python examples/run_verification.py

Outputs under ``docs/results/``:
- ``taylor_green_verification.png/.svg``  cover image: fields, normalized
  diagnostics, and time-convergence of the Taylor-Green verification case
- ``nonlinear_cascade.png/.svg``           computed nonlinear demonstration
- ``vorticity_evolution.gif``              animated nonlinear vorticity field
- ``project_preview.png``                  1280x640 flagship preview
- ``taylor_green_history.csv``             time, energies, enstrophy, error
- ``time_convergence.csv``                 dt sweep with observed order
- ``nonlinear_history.csv``, ``nonlinear_spectra.csv``

Presentation conventions: white background, no grid lines, physical
coordinates in meters-equivalent domain units, transposed fields so rows
map to y and columns to x (the solver uses ``indexing='ij'``).
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spectral_flow import SpectralVorticitySolver, taylor_green_vorticity
from spectral_flow.solver import taylor_green_velocity

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, RED, GRAY = "#1f4e79", "#0b6e4f", "#b23a48", "#444444"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def grid(n):
    x = np.arange(n) * (2.0 * np.pi / n)
    return x, np.meshgrid(x, x, indexing="ij")


def shell_spectrum(solver, w_hat):
    """Shell-summed discrete energy statistic E(k).

    Sum of 0.5*(|u_hat|^2 + |v_hat|^2)/N^4 over all discrete modes with
    rounded |k| = k. By Parseval the total sum_k E(k) equals the mean
    kinetic energy of the recovered velocity field; this is a discrete
    statistic of the retained Fourier coefficients, not a continuous
    spectral density.
    """
    u_hat, v_hat = solver.velocity_hat(w_hat)
    energy_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / solver.n**4
    kr = np.rint(np.sqrt(solver.k2)).astype(int)
    return np.bincount(kr.ravel(), weights=energy_hat.ravel())


# ----------------------------------------------------------------------
# 1. Taylor-Green verification (n=128, nu=0.01)
# ----------------------------------------------------------------------
def run_taylor_green():
    n, nu, dt, t_end = 128, 0.01, 0.001, 2.0
    solver = SpectralVorticitySolver(n=n, viscosity=nu)
    x, (xx, yy) = grid(n)
    w0 = taylor_green_vorticity(xx, yy)

    sample = 50  # every 0.05 s
    history = solver.integrate(w0, dt=dt, nsteps=int(t_end / dt), sample_every=sample)

    rows = ["time,E_numeric,E_exact,Z_numeric,Z_exact,L2_velocity_error"]
    for t, w_hat in history:
        f = np.exp(-2.0 * nu * t)
        rows.append(
            f"{t:.6f},{solver.energy(w_hat):.12e},{0.25 * f * f:.12e},"
            f"{solver.enstrophy(w_hat):.12e},{0.5 * f * f:.12e},"
            f"{solver.velocity_l2_error(w_hat, t):.12e}"
        )
    (RESULTS / "taylor_green_history.csv").write_text("\n".join(rows), encoding="utf-8")

    t = np.array([h[0] for h in history])
    E = np.array([solver.energy(h[1]) for h in history])
    Z = np.array([solver.enstrophy(h[1]) for h in history])
    E_exact = 0.25 * np.exp(-4.0 * nu * t)
    Z_exact = 0.50 * np.exp(-4.0 * nu * t)
    err = np.array([solver.velocity_l2_error(h[1], tt) for tt, h in zip(t, history)])

    w0_hat = np.fft.fft2(w0)
    wT_hat = history[-1][1]
    w0_field = np.fft.ifft2(w0_hat).real
    wT_field = np.fft.ifft2(wT_hat).real
    u_num, v_num = solver.velocity(wT_hat)
    u_ex, v_ex = taylor_green_velocity(xx, yy, t_end, nu)
    err_field = np.sqrt((u_num - u_ex) ** 2 + (v_num - v_ex) ** 2)

    return solver, x, (t, E, E_exact, Z, Z_exact, err), (w0_field, wT_field, err_field)


def run_convergence():
    """dt sweep in the strongly damped regime (matches the test suite)."""
    solver = SpectralVorticitySolver(n=8, viscosity=0.5)
    _, (xx, yy) = grid(8)
    w0 = taylor_green_vorticity(xx, yy)
    dts, errors = [], []
    for dt in (0.2, 0.1, 0.05, 0.025):
        nsteps = int(round(1.0 / dt))
        _, w_hat = solver.integrate(w0, dt=dt, nsteps=nsteps)[-1]
        dts.append(dt)
        errors.append(solver.velocity_l2_error(w_hat, nsteps * dt))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    rows = ["dt,L2_velocity_error"] + [f"{d},{e:.12e}" for d, e in zip(dts, errors)]
    rows.append("")
    rows += [f"# pair orders: {', '.join(f'{o:.3f}' for o in orders)}"]
    (RESULTS / "time_convergence.csv").write_text("\n".join(rows), encoding="utf-8")
    return np.array(dts), np.array(errors), orders


# ----------------------------------------------------------------------
# 2. Nonlinear cascade demonstration (computed, no external reference)
# ----------------------------------------------------------------------
def run_cascade():
    n, nu, dt, t_end = 128, 5.0e-4, 0.005, 8.0
    solver = SpectralVorticitySolver(n=n, viscosity=nu)
    x, (xx, yy) = grid(n)
    rng = np.random.default_rng(20261018)
    # Band-limited large-scale vortices plus a small filtered perturbation:
    # everything the dynamics develop afterwards comes from the computed run.
    w0 = np.sin(xx) * np.sin(yy) + 0.5 * np.cos(2 * xx) * np.cos(2 * yy)
    noise_hat = np.fft.fft2(rng.standard_normal((n, n))) * solver.dealias_mask
    noise_hat[np.sqrt(solver.k2) > 6] = 0.0
    w0 = w0 + 0.05 * np.fft.ifft2(noise_hat).real

    sample = 20  # dense sampling for the animation
    history = solver.integrate(w0, dt=dt, nsteps=int(t_end / dt), sample_every=sample)

    rows = ["time,energy,enstrophy"]
    for t, w_hat in history:
        rows.append(f"{t:.6f},{solver.energy(w_hat):.12e},{solver.enstrophy(w_hat):.12e}")
    (RESULTS / "nonlinear_history.csv").write_text("\n".join(rows), encoding="utf-8")

    t = np.array([h[0] for h in history])
    E = np.array([solver.energy(h[1]) for h in history])
    Z = np.array([solver.enstrophy(h[1]) for h in history])

    snaps, spectra = [], []
    for t_target in (0.0, 4.0, 8.0):
        idx = int(np.argmin(np.abs(t - t_target)))
        w_hat = history[idx][1]
        snaps.append(np.fft.ifft2(w_hat).real)
        spectra.append(shell_spectrum(solver, w_hat))

    spec_rows = ["k,E_k_t0,E_k_t4,E_k_t8"]
    for k in range(1, min(len(s) for s in spectra)):
        spec_rows.append(f"{k},{spectra[0][k]:.12e},{spectra[1][k]:.12e},{spectra[2][k]:.12e}")
    (RESULTS / "nonlinear_spectra.csv").write_text("\n".join(spec_rows), encoding="utf-8")

    return solver, x, (t, E, Z), snaps, spectra, history


# ----------------------------------------------------------------------
# Figures (white background, no grid lines, physical coordinates)
# ----------------------------------------------------------------------
def make_cover(solver, x, diag, fields, conv):
    t, E, E_exact, Z, Z_exact, err = diag
    w0_field, wT_field, err_field = fields
    dts, errors, orders = conv

    fig = plt.figure(figsize=(13.4, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.15, 1.0), hspace=0.38, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    pm = ax0.pcolormesh(x, x, w0_field.T, cmap="RdBu_r", shading="nearest", rasterized=True)
    ax0.set_title(r"$\omega$ at $t=0$ s")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    fig.colorbar(pm, ax=ax0, label=r"$\omega$ [1/s]")

    ax1 = fig.add_subplot(gs[0, 1])
    pm = ax1.pcolormesh(x, x, wT_field.T, cmap="RdBu_r", shading="nearest", rasterized=True)
    ax1.set_title(r"$\omega$ at $t=2$ s, $\nu=0.01$")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.colorbar(pm, ax=ax1, label=r"$\omega$ [1/s]")

    ax2 = fig.add_subplot(gs[0, 2])
    pm = ax2.pcolormesh(x, x, err_field.T, cmap="magma", shading="nearest", rasterized=True)
    ax2.set_title(f"velocity error at t = 2 s\nmax = {err_field.max():.2e} m/s")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    fig.colorbar(pm, ax=ax2, label="error [m/s]")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, E / E[0], "o", ms=3.5, color=BLUE, label="E / E(0) numeric")
    ax3.plot(t, E_exact / E_exact[0], "-", color=BLUE, lw=1.2, label="E / E(0) exact")
    ax3.plot(t, Z / Z[0], "s", ms=3.5, color=RED, label="Z / Z(0) numeric")
    ax3.plot(t, Z_exact / Z_exact[0], "-", color=RED, lw=1.2, label="Z / Z(0) exact")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("normalized energy / enstrophy [-]")
    ax3.set_title("Energy and enstrophy vs analytic decay\n(dimensionless, normalized)")
    ax3.legend(fontsize=8, ncol=2, frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t, err, ".", ms=3.5, color=GREEN)
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("relative L2 velocity error [-]")
    ax4.set_title(f"Error history (max {err.max():.1e}; roundoff floor)")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.loglog(dts, errors, "o-", color=RED, label="RK4 measured")
    ref = errors[0] * (dts / dts[0]) ** 4
    ax5.loglog(dts, ref, "--", color=GRAY, label=r"$\mathcal{O}(\Delta t^4)$")
    ax5.set_xticks(dts)
    ax5.set_xticklabels([f"{d:g}" for d in dts])
    ax5.minorticks_off()
    ax5.set_xlabel(r"$\Delta t$ [s]")
    ax5.set_ylabel("relative L2 error at t = 1 s [-]")
    ax5.set_title(f"Time convergence (N=8, ν=0.5)\nmeasured order ≈ {orders.mean():.2f}")
    ax5.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Vortex Notebook — Taylor-Green case: velocity error at the roundoff floor "
        f"({err.max():.1e}), RK4 order ≈ {orders.mean():.2f}",
        fontsize=13,
    )
    fig.savefig(RESULTS / "taylor_green_verification.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "taylor_green_verification.svg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_cascade_figure(solver, x, hist, snaps, spectra):
    t, E, Z = hist
    fig, ax = plt.subplots(2, 3, figsize=(13.4, 8.4))
    for a, w, tt in zip(ax[0], snaps, ("0", "4", "8")):
        pm = a.pcolormesh(x, x, w.T, cmap="RdBu_r", shading="nearest", rasterized=True)
        a.set_aspect("equal")
        a.set_title(f"ω at t = {tt} s")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        fig.colorbar(pm, ax=a, label=r"$\omega$ [1/s]")

    ax[1, 0].plot(t, E / E[0], color=BLUE, lw=1.6)
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("E / E(0) [-]")
    ax[1, 0].set_title("Kinetic energy (ν = 5e-4 m²/s)")

    ax[1, 1].plot(t, Z / Z[0], color=RED, lw=1.6)
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("Z / Z(0) [-]")
    ax[1, 1].set_title("Enstrophy decay (cascade dissipation)")

    kk = np.arange(len(spectra[0]))
    for spec, tt, color in zip(spectra, ("0", "4", "8"), (BLUE, "#8a6d1f", RED)):
        pos = spec > 0
        ax[1, 2].loglog(kk[pos], spec[pos], "o-", ms=2.5, lw=1.0, color=color, label=f"t = {tt} s")
    ax[1, 2].set_xlabel("shell index |k| [-]")
    ax[1, 2].set_ylabel("energy in shell E(k) [m²/s²]")
    ax[1, 2].set_title("Shell-summed energy statistic\n(Σ_k E(k) = mean kinetic energy)")
    ax[1, 2].legend(fontsize=8, frameon=False)

    fig.suptitle("Nonlinear cascade demonstration — computed run, no external reference", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "nonlinear_cascade.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "nonlinear_cascade.svg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_gif(solver, x, history, stride=2, fps=8):
    """Animate the computed vorticity field of the nonlinear run."""
    frames = history[::stride]
    vmax = max(np.abs(np.fft.ifft2(w_hat)).max() for _, w_hat in frames)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=140)
    (pm,) = [ax.pcolormesh(x, x, np.fft.ifft2(frames[0][1]).real.T,
                           cmap="RdBu_r", shading="nearest", vmin=-vmax, vmax=vmax)]
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("Vorticity ω — computed nonlinear run (t = 0.00 s)")
    fig.colorbar(pm, ax=ax, label=r"$\omega$ [1/s]")

    writer = mpl_animation.PillowWriter(fps=fps)
    with writer.saving(fig, RESULTS / "vorticity_evolution.gif", dpi=140):
        for t, w_hat in frames:
            field = np.fft.ifft2(w_hat).real
            pm.set_array(field.T.ravel())
            title.set_text(f"Vorticity ω — computed nonlinear run (t = {t:.2f} s)")
            writer.grab_frame()
    plt.close(fig)
    return len(frames)


def make_preview(diag, fields, conv, x):
    t, E, E_exact, Z, Z_exact, err = diag
    _, wT_field, _ = fields
    dts, errors, orders = conv

    fig, ax = plt.subplots(1, 3, figsize=(12.8, 6.4))
    pm = ax[0].pcolormesh(x, x, wT_field.T, cmap="RdBu_r", shading="nearest", rasterized=True)
    ax[0].set_aspect("equal")
    ax[0].set_title("Computed vorticity ω at t = 2 s")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    fig.colorbar(pm, ax=ax[0], shrink=0.82, label=r"$\omega$ [1/s]")

    ax[1].plot(t, E / E[0], "o", ms=3, color=BLUE, label="numeric")
    ax[1].plot(t, E_exact / E_exact[0], "-", color=RED, label="analytic")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("E / E(0) [-]")
    ax[1].set_title("Kinetic energy vs exact decay")
    ax[1].legend(frameon=False, fontsize=9)

    ax[2].loglog(dts, errors, "o-", color=RED)
    ax[2].loglog(dts, errors[0] * (dts / dts[0]) ** 4, "--", color=GRAY)
    ax[2].set_xticks(dts)
    ax[2].set_xticklabels([f"{d:g}" for d in dts])
    ax[2].minorticks_off()
    ax[2].set_xlabel(r"$\Delta t$ [s]")
    ax[2].set_ylabel("relative L2 error [-]")
    ax[2].set_title(f"RK4 convergence, order ≈ {orders.mean():.1f}")

    fig.suptitle(
        "Vortex Notebook — Taylor-Green case error at the roundoff floor "
        f"({err.max():.0e}); RK4 order {orders.mean():.2f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "project_preview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    solver, x, diag, fields = run_taylor_green()
    conv = run_convergence()
    cas_solver, cas_x, hist, snaps, spectra, history = run_cascade()

    make_cover(solver, x, diag, fields, conv)
    make_cascade_figure(cas_solver, cas_x, hist, snaps, spectra)
    n_frames = make_gif(cas_solver, cas_x, history)
    make_preview(diag, fields, conv, x)

    t, E, E_exact, Z, Z_exact, err = diag
    _, _, orders = conv
    summary = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.platform(),
        },
        "taylor_green": {
            "grid": 128,
            "viscosity": 0.01,
            "max_relative_L2_velocity_error": float(err.max()),
            "final_relative_L2_velocity_error": float(err[-1]),
        },
        "time_convergence": {
            "regime": "nu=0.5, n=8, t=1.0 (matches test suite)",
            "observed_orders": [float(o) for o in orders],
        },
        "nonlinear": {"grid": 128, "viscosity": 5.0e-4, "t_end": 8.0},
        "animation": {"file": "vorticity_evolution.gif", "frames": n_frames},
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
