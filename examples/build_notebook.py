"""Build and (with --execute) run docs/vortex_notebook.ipynb.

The notebook is the executable guide for this repository: markdown
derivations plus live computations of every verification case. Regenerate
after editing with:

    python examples/build_notebook.py --execute            # portable kernel
    python examples/build_notebook.py --execute --kernel engineering-python

The default kernel is the portable ``python3`` kernelspec that any normal
``pip install ipykernel`` environment provides; ``--kernel`` overrides it
for environments with a differently named kernel. Execution runs with the
repository root as the working directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]


def code(src):
    return nbf.v4.new_code_cell(src)


def md(src):
    return nbf.v4.new_markdown_cell(src)


def build(kernel: str = "python3"):
    nb = nbf.v4.new_notebook()
    display = {"python3": "Python 3 (ipykernel)"}.get(kernel, kernel)
    nb.metadata.kernelspec = {
        "display_name": display,
        "language": "python",
        "name": kernel,
    }
    nb.metadata.language_info = {"name": "python"}

    cells = [
        md(
            "# Vortex Notebook — an executable guide to periodic vortex dynamics\n"
            "\n"
            "This notebook drives the tested `spectral_flow` package: it derives the\n"
            "method, runs the Taylor–Green verification case against the analytic\n"
            "solution, measures the RK4 time-convergence order, and evolves a nonlinear\n"
            "cascade. The plots in the code cells are computed live when the notebook\n"
            "runs; the closing animation is loaded from the higher-resolution recorded\n"
            "run produced by `examples/run_verification.py` (not re-computed here).\n"
            "\n"
            "Run from the repository root with the dependencies from\n"
            "`requirements-dev.txt` installed (any environment with an `ipykernel`\n"
            "`python3` kernel works):\n"
            "\n"
            "```bash\n"
            "python -m ipykernel install --user --name python3  # if no kernel yet\n"
            "python examples/build_notebook.py --execute\n"
            "```\n"
            "\n"
            "Static export of the executed copy committed here is rendered directly\n"
            "on GitHub."
        ),
        md(
            "## 1. Model and method\n"
            "\n"
            "For velocity $\\mathbf{u}=(u,v)$, vorticity $\\omega=\\partial_x v-\\partial_y u$,\n"
            "and kinematic viscosity $\\nu$ on the periodic square $[0,2\\pi)^2$:\n"
            "\n"
            "$$\\frac{\\partial\\omega}{\\partial t} + \\nabla\\cdot(\\mathbf{u}\\,\\omega) = \\nu\\nabla^2\\omega,\n"
            "\\qquad \\hat{\\psi}=\\frac{\\hat{\\omega}}{k^2}\\ (k\\neq 0),\\quad\n"
            "\\hat{u}=ik_y\\hat{\\psi},\\ \\hat{v}=-ik_x\\hat{\\psi}.$$\n"
            "\n"
            "The sign matters: $\\omega=-\\nabla^2\\psi$ gives $\\hat\\psi=+\\hat\\omega/k^2$.\n"
            "The nonlinear term is evaluated in conservative form and both it and the\n"
            "solution are projected onto the strict band $|k_x|,|k_y|<N/3$, which makes\n"
            "quadratic products alias-free on the retained band (an inclusive $N/3$\n"
            "cutoff would fold boundary pairs back into the band). Time stepping is\n"
            "classical RK4. The mean vorticity $\\hat\\omega(0,0)$ is an exactly conserved\n"
            "passive constant; a periodic streamfunction cannot represent a uniform\n"
            "background curl, so the recovered velocity is the curl-inverse of the\n"
            "zero-mean part of $\\omega$."
        ),
        code(
            "import pathlib, sys\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "NB_DIR = pathlib.Path('docs') if pathlib.Path('docs').exists() else pathlib.Path('.')\n"
            "sys.path.insert(0, str(NB_DIR.resolve().parent))\n"
            "\n"
            "from spectral_flow import SpectralVorticitySolver, taylor_green_vorticity\n"
            "from spectral_flow.solver import taylor_green_velocity\n"
            "\n"
            "plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',\n"
            "                     'axes.grid': False, 'axes.spines.top': False,\n"
            "                     'axes.spines.right': False})\n"
            "print('numpy', np.__version__)"
        ),
        md(
            "## 2. Taylor–Green verification\n"
            "\n"
            "The Taylor–Green vortex\n"
            "$u=-\\cos x\\sin y\\,f$, $v=\\sin x\\cos y\\,f$, $\\omega=2\\cos x\\cos y\\,f$\n"
            "with $f=e^{-2\\nu t}$ is an exact solution (its nonlinear term vanishes\n"
            "identically), with $E(t)=\\tfrac14 e^{-4\\nu t}$ and $Z(t)=\\tfrac12 e^{-4\\nu t}$.\n"
            "It therefore verifies the Poisson inversion, the viscous operator, and the\n"
            "time integrator simultaneously."
        ),
        code(
            "n, nu, dt, t_end = 64, 0.01, 0.001, 1.0\n"
            "solver = SpectralVorticitySolver(n=n, viscosity=nu)\n"
            "x = np.arange(n) * (2*np.pi/n)\n"
            "xx, yy = np.meshgrid(x, x, indexing='ij')\n"
            "history = solver.integrate(taylor_green_vorticity(xx, yy), dt=dt,\n"
            "                           nsteps=int(t_end/dt), sample_every=50)\n"
            "t = np.array([h[0] for h in history])\n"
            "E = np.array([solver.energy(h[1]) for h in history])\n"
            "Z = np.array([solver.enstrophy(h[1]) for h in history])\n"
            "err = np.array([solver.velocity_l2_error(h[1], tt) for tt, h in zip(t, history)])\n"
            "E_exact = 0.25*np.exp(-4*nu*t); Z_exact = 0.5*np.exp(-4*nu*t)\n"
            "\n"
            "wT = np.fft.ifft2(history[-1][1]).real\n"
            "u_num, v_num = solver.velocity(history[-1][1])\n"
            "u_ex, v_ex = taylor_green_velocity(xx, yy, t_end, nu)\n"
            "err_field = np.sqrt((u_num-u_ex)**2 + (v_num-v_ex)**2)\n"
            "\n"
            "fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.0))\n"
            "for a, field, title, cmap in ((ax[0], wT, 'vorticity ω at t = 1 s', 'RdBu_r'),\n"
            "                              (ax[1], err_field, 'velocity error |u_num − u_exact|', 'magma')):\n"
            "    pm = a.pcolormesh(x, x, field.T, cmap=cmap, shading='nearest')\n"
            "    a.set_aspect('equal'); a.set_xlabel('x [m]'); a.set_ylabel('y [m]')\n"
            "    a.set_title(title)\n"
            "    fig.colorbar(pm, ax=a, shrink=0.85)\n"
            "ax[2].semilogy(t, err, '.', color='#0b6e4f')\n"
            "ax[2].set_xlabel('t [s]'); ax[2].set_ylabel('relative L2 velocity error [-]')\n"
            "ax[2].set_title(f'error history (max {err.max():.1e})')\n"
            "fig.tight_layout(); plt.show()\n"
            "print(f'max relative L2 velocity error: {err.max():.3e}')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6.4, 4.0))\n"
            "ax.plot(t, E/E[0], 'o', ms=3, color='#1f4e79', label='E/E(0) numeric')\n"
            "ax.plot(t, E_exact/E_exact[0], '-', color='#1f4e79', lw=1.2, label='E/E(0) exact')\n"
            "ax.plot(t, Z/Z[0], 's', ms=3, color='#b23a48', label='Z/Z(0) numeric')\n"
            "ax.plot(t, Z_exact/Z_exact[0], '-', color='#b23a48', lw=1.2, label='Z/Z(0) exact')\n"
            "ax.set_xlabel('t [s]'); ax.set_ylabel('normalized energy / enstrophy [-]')\n"
            "ax.legend(frameon=False, fontsize=8)\n"
            "fig.tight_layout(); plt.show()\n"
            "print(f'energy rel. deviation from exact: {abs(E[-1]/E_exact[-1]-1):.2e}')\n"
            "print(f'enstrophy rel. deviation from exact: {abs(Z[-1]/Z_exact[-1]-1):.2e}')"
        ),
        md(
            "## 3. RK4 time-convergence order\n"
            "\n"
            "At $\\nu=0.01$ the RK4 truncation error is far below the floating-point\n"
            "roundoff floor, so the order must be measured in a strongly damped regime\n"
            "($N=8$, $\\nu=0.5$, $T=1$ s) where the temporal error is resolvable; the\n"
            "finest step $\\Delta t=0.025$ keeps the error near $10^{-9}$, well above\n"
            "roundoff, so no saturation contaminates the slope."
        ),
        code(
            "conv_solver = SpectralVorticitySolver(n=8, viscosity=0.5)\n"
            "w0c = taylor_green_vorticity(*np.meshgrid(np.arange(8)*(2*np.pi/8), np.arange(8)*(2*np.pi/8), indexing='ij'))\n"
            "dts, errors = [], []\n"
            "for dt in (0.2, 0.1, 0.05, 0.025):\n"
            "    nsteps = int(round(1.0/dt))\n"
            "    _, wh = conv_solver.integrate(w0c, dt=dt, nsteps=nsteps)[-1]\n"
            "    dts.append(dt); errors.append(conv_solver.velocity_l2_error(wh, 1.0))\n"
            "orders = np.log2(np.array(errors[:-1])/np.array(errors[1:]))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(5.6, 4.0))\n"
            "ax.loglog(dts, errors, 'o-', color='#b23a48', label='RK4 measured')\n"
            "ax.loglog(dts, errors[0]*(np.array(dts)/dts[0])**4, '--', color='#444444', label='O(Δt⁴)')\n"
            "ax.set_xticks(dts); ax.set_xticklabels([f'{d:g}' for d in dts]); ax.minorticks_off()\n"
            "ax.set_xlabel('Δt [s]'); ax.set_ylabel('relative L2 error at t = 1 s [-]')\n"
            "ax.legend(frameon=False, fontsize=8)\n"
            "fig.tight_layout(); plt.show()\n"
            "print('measured orders:', np.round(orders, 3))"
        ),
        md(
            "## 4. Nonlinear cascade (computed demonstration)\n"
            "\n"
            "A band-limited large-scale initial condition plus small filtered noise\n"
            "(seed 20261018) is evolved at low viscosity. Vortices merge, enstrophy\n"
            "decays through the cascade, and the shell-summed energy statistic\n"
            "$E(k)=\\sum_{|k'|=k} \\tfrac12(|\\hat u|^2+|\\hat v|^2)/N^4$ broadens toward\n"
            "higher shells. By Parseval, $\\sum_k E(k)$ equals the mean kinetic energy —\n"
            "checked below. **There is no external reference for this case and none is\n"
            "claimed**; it demonstrates the nonlinear machinery on a computed run."
        ),
        code(
            "n2, nu2, dt2, t_end2 = 96, 5e-4, 0.005, 6.0\n"
            "cas = SpectralVorticitySolver(n=n2, viscosity=nu2)\n"
            "x2 = np.arange(n2)*(2*np.pi/n2)\n"
            "xx2, yy2 = np.meshgrid(x2, x2, indexing='ij')\n"
            "rng = np.random.default_rng(20261018)\n"
            "w0 = np.sin(xx2)*np.sin(yy2) + 0.5*np.cos(2*xx2)*np.cos(2*yy2)\n"
            "noise = np.fft.fft2(rng.standard_normal((n2, n2))) * cas.dealias_mask\n"
            "noise[np.sqrt(cas.k2) > 6] = 0.0\n"
            "w0 = w0 + 0.05*np.fft.ifft2(noise).real\n"
            "hist2 = cas.integrate(w0, dt=dt2, nsteps=int(t_end2/dt2), sample_every=100)\n"
            "t2 = np.array([h[0] for h in hist2])\n"
            "E2 = np.array([cas.energy(h[1]) for h in hist2])\n"
            "Z2 = np.array([cas.enstrophy(h[1]) for h in hist2])\n"
            "\n"
            "def shell_energy(wh):\n"
            "    uh, vh = cas.velocity_hat(wh)\n"
            "    e = 0.5*(np.abs(uh)**2 + np.abs(vh)**2)/n2**4\n"
            "    kr = np.rint(np.sqrt(cas.k2)).astype(int)\n"
            "    return np.bincount(kr.ravel(), weights=e.ravel())\n"
            "\n"
            "snaps = [np.fft.ifft2(hist2[i][1]).real for i in (0, len(hist2)//2, -1)]\n"
            "specs = [shell_energy(hist2[i][1]) for i in (0, len(hist2)//2, -1)]\n"
            "parseval_gap = abs(specs[-1].sum() - E2[-1]) / E2[-1]\n"
            "\n"
            "fig, ax = plt.subplots(2, 3, figsize=(12.6, 7.6))\n"
            "for a, w, tt in zip(ax[0], snaps, ('0', '3', '6')):\n"
            "    pm = a.pcolormesh(x2, x2, w.T, cmap='RdBu_r', shading='nearest')\n"
            "    a.set_aspect('equal'); a.set_xlabel('x [m]'); a.set_ylabel('y [m]')\n"
            "    a.set_title(f'ω at t = {tt} s')\n"
            "    fig.colorbar(pm, ax=a, shrink=0.85, label='ω [1/s]')\n"
            "ax[1,0].plot(t2, E2/E2[0], color='#1f4e79'); ax[1,0].set_xlabel('t [s]')\n"
            "ax[1,0].set_ylabel('E/E(0) [-]'); ax[1,0].set_title('kinetic energy')\n"
            "ax[1,1].plot(t2, Z2/Z2[0], color='#b23a48'); ax[1,1].set_xlabel('t [s]')\n"
            "ax[1,1].set_ylabel('Z/Z(0) [-]'); ax[1,1].set_title('enstrophy decay')\n"
            "kk = np.arange(len(specs[-1]))\n"
            "for spec, tt, c in zip(specs, ('0','3','6'), ('#1f4e79', '#8a6d1f', '#b23a48')):\n"
            "    pos = spec > 0\n"
            "    ax[1,2].loglog(kk[pos], spec[pos], 'o-', ms=2.5, lw=1.0, color=c, label=f't={tt} s')\n"
            "ax[1,2].set_xlabel('shell index |k| [-]'); ax[1,2].set_ylabel('E(k) [m²/s²]')\n"
            "ax[1,2].set_title('shell-summed energy statistic'); ax[1,2].legend(frameon=False, fontsize=8)\n"
            "fig.tight_layout(); plt.show()\n"
            "print(f'Parseval check: |Σ_k E(k) − E_mean| / E_mean = {parseval_gap:.2e}')"
        ),
        md(
            "## 5. Recorded vorticity animation\n"
            "\n"
            "The full-resolution (N=128, 8 s, 41-frame) animation produced by\n"
            "`examples/run_verification.py` (the committed reproduction script):"
        ),
        code(
            "from IPython.display import Image\n"
            "Image(filename=str(NB_DIR / 'results' / 'vorticity_evolution.gif'))"
        ),
        md(
            "## 6. Explore your own parameters\n"
            "\n"
            "Edit and re-run the next cell: resolution `n_exp`, viscosity `nu_exp`,\n"
            "end time `t_exp`, and any initial vorticity field. Keep the explicit\n"
            "stability bounds in mind: $\\nu\\,2(N/2)^2\\Delta t \\lesssim 2.8$ and\n"
            "$\\max|u|\\,\\Delta t/\\Delta x \\lesssim 1$."
        ),
        code(
            "n_exp, nu_exp, t_exp = 96, 2e-3, 3.0\n"
            "dt_exp = 0.002\n"
            "exp_solver = SpectralVorticitySolver(n=n_exp, viscosity=nu_exp)\n"
            "xe = np.arange(n_exp)*(2*np.pi/n_exp)\n"
            "xxe, yye = np.meshgrid(xe, xe, indexing='ij')\n"
            "w0_exp = np.sin(2*xxe)*np.cos(yye) + 0.4*np.sin(xxe)*np.sin(3*yye)\n"
            "hexp = exp_solver.integrate(w0_exp, dt=dt_exp, nsteps=int(t_exp/dt_exp), sample_every=200)\n"
            "fig, ax = plt.subplots(figsize=(5.6, 4.4))\n"
            "pm = ax.pcolormesh(xe, xe, np.fft.ifft2(hexp[-1][1]).real.T, cmap='RdBu_r', shading='nearest')\n"
            "ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')\n"
            "ax.set_title(f'your run: ω at t = {hexp[-1][0]:.1f} s')\n"
            "fig.colorbar(pm, ax=ax, shrink=0.85, label='ω [1/s]')\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## 7. What this establishes — and what it does not\n"
            "\n"
            "Verified here: exact analytic Taylor–Green decay to the roundoff floor at\n"
            "the tested parameters, RK4 order ≈ 4 in a resolvable regime, Parseval\n"
            "closure of the shell statistic, and a stable computed nonlinear cascade.\n"
            "The test suite adds machine-precision divergence freedom, curl round trips,\n"
            "and an independent 3/2-rule padded-reference alias check.\n"
            "\n"
            "Not established: turbulence-statistics validation, high-Reynolds studies,\n"
            "or agreement with experiments. The Taylor–Green case verifies the linear\n"
            "machinery exactly (its nonlinear term is identically zero); nonlinear\n"
            "correctness rests on the padded reference, not on an external\n"
            "turbulent-flow benchmark. See `docs/verification.md` for details."
        ),
    ]
    nb.cells = cells
    out = ROOT / "docs" / "vortex_notebook.ipynb"
    nbf.write(nb, str(out))
    print(f"wrote {out}")
    return out


def execute(path, kernel):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name=kernel,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()  # real execution errors propagate
    nbformat.write(nb, str(path))
    print(f"executed in place with kernel {kernel!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: portable 'python3')",
    )
    args = parser.parse_args()
    path = build(args.kernel)
    if args.execute:
        execute(path, args.kernel)
        print("executed in place")
