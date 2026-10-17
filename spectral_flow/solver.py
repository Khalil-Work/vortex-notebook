"""Pseudospectral vorticity-streamfunction solver for the 2D Navier-Stokes
equations on a doubly periodic square domain.

Governing equations
-------------------
With velocity :math:`\\mathbf{u}=(u,v)`, vorticity
:math:`\\omega=\\partial_x v-\\partial_y u`, and kinematic viscosity
:math:`\\nu`, the 2D incompressible Navier-Stokes equations in the
vorticity-streamfunction form read

.. math::

    \\frac{\\partial\\omega}{\\partial t}
        + \\nabla\\cdot(\\mathbf{u}\\,\\omega) = \\nu\\nabla^2\\omega,
    \\qquad
    \\hat{\\psi} = -\\frac{\\hat{\\omega}}{k^2}\\ (k\\neq 0),\\quad
    \\hat{\\psi}(0)=0,

.. math::

    \\hat{u} = i k_y \\hat{\\psi}, \\qquad
    \\hat{v} = -i k_x \\hat{\\psi}.

The nonlinear term is evaluated in conservative (divergence) form: the
products :math:`u\\omega` and :math:`v\\omega` are formed in physical space,
transformed, differentiated spectrally, and the result is truncated to the
retained band. Both the nonlinear term output and the solution itself are
projected onto the band :math:`|k_x|, |k_y| < N/3` (strictly). With this
choice the discrete quadratic products are alias-free on the retained band:
two retained modes can only fold back onto a retained mode if their true sum
also lies there, because :math:`2(N/3 - 1) - N < -(N/3)`. The scheme is thus
an exact Galerkin integration of the Navier-Stokes equations restricted to
that band. Time stepping is classical four-stage Runge-Kutta (RK4).

The Poisson relation :math:`\\omega=-\\nabla^2\\psi` is inverted in Fourier
space as :math:`\\hat{\\psi}=\\hat{\\omega}/k^2`. The streamfunction zero
mode fixes the arbitrary additive constant of :math:`\\psi` and carries no
velocity. Mean-vorticity behavior on the periodic domain: the right-hand
side has an identically zero :math:`k=0` coefficient (the spectral
divergence of anything has no mean and the viscous term vanishes at
:math:`k=0`), so :math:`\\hat{\\omega}(0,0)` is an exactly conserved passive
constant of the discrete scheme. Because a periodic single-valued
streamfunction cannot represent a uniform background curl, the recovered
velocity is the curl-inverse of the *zero-mean* part of :math:`\\omega`; the
curl round trip returns :math:`\\omega` with its mean removed. For the
2D periodic problems targeted here the mean does not participate in the
dynamics.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpectralVorticitySolver", "taylor_green_vorticity"]


def _wavenumbers(n: int):
    """Integer FFT wavenumbers for ``n`` points on [0, 2*pi).

    On this domain the Fourier modes are exp(i j x) with integer j, so the
    spectral derivative operator is simply multiplication by i*j.
    """
    return np.fft.fftfreq(n, d=1.0 / n)


def taylor_green_vorticity(x, y):
    """Initial vorticity of the Taylor-Green vortex, ``2 cos x cos y``.

    The corresponding exact velocity field is
    ``u = -cos x sin y f``, ``v = sin x cos y f`` with the decay factor
    ``f = exp(-2 nu t)`` for the Navier-Stokes equations (the nonlinear term
    vanishes identically for this single-mode field).
    """
    return 2.0 * np.cos(x) * np.cos(y)


def taylor_green_velocity(x, y, t, nu):
    """Exact Taylor-Green velocity ``(u, v)`` at time ``t``."""
    f = np.exp(-2.0 * nu * t)
    return -np.cos(x) * np.sin(y) * f, np.sin(x) * np.cos(y) * f


class SpectralVorticitySolver:
    """Dealiased pseudospectral solver on :math:`[0,2\\pi)^2`.

    Parameters
    ----------
    n : int
        Number of grid points per direction (``n`` even).
    viscosity : float
        Kinematic viscosity :math:`\\nu > 0`.
    dealias : bool
        Apply the 2/3-rule truncation to the nonlinear term (recommended).
    """

    def __init__(self, n: int, viscosity: float, dealias: bool = True):
        if n < 8 or n % 2 != 0:
            raise ValueError("n must be an even integer >= 8")
        if viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        self.n = n
        self.nu = viscosity
        self.dealias = dealias

        k = _wavenumbers(n)
        self.kx = np.meshgrid(k, k, indexing="ij")[0]
        self.ky = np.meshgrid(k, k, indexing="ij")[1]
        self.k2 = self.kx**2 + self.ky**2

        # Poisson relation omega = -laplacium(psi) gives omega_hat = k^2
        # psi_hat, so psi_hat = omega_hat / k^2 with the zero mode pinned to
        # zero: psi determines velocities only up to a constant, and the
        # constant carries no velocity.
        inv_k2 = np.zeros_like(self.k2)
        nonzero = self.k2 > 0.0
        inv_k2[nonzero] = 1.0 / self.k2[nonzero]
        self._psi_multiplier = inv_k2

        # Retained band: |k| < N/3 STRICTLY per direction. An inclusive
        # N/3 boundary would let boundary pairs fold back into the band
        # (e.g. on N=48 the mode 16 pair sums to 32, aliasing to -16).
        kcut = n / 3.0
        self.dealias_mask = (np.abs(self.kx) < kcut) & (np.abs(self.ky) < kcut)

        # Viscous multiplier for the linear term.
        self._viscous = -viscosity * self.k2

    # ------------------------------------------------------------------
    # Spectral operators
    # ------------------------------------------------------------------

    def streamfunction_hat(self, w_hat):
        """Return :math:`\\hat{\\psi}` from :math:`\\hat{\\omega}`."""
        return self._psi_multiplier * w_hat

    def velocity(self, w_hat):
        """Physical velocities ``(u, v)`` from spectral vorticity."""
        psi_hat = self.streamfunction_hat(w_hat)
        u = np.fft.ifft2(1j * self.ky * psi_hat).real
        v = np.fft.ifft2(-1j * self.kx * psi_hat).real
        return u, v

    def velocity_hat(self, w_hat):
        """Spectral velocities ``(u_hat, v_hat)`` from spectral vorticity."""
        psi_hat = self.streamfunction_hat(w_hat)
        return 1j * self.ky * psi_hat, -1j * self.kx * psi_hat

    def nonlinear_hat(self, w_hat):
        """Spectral :math:`-\\nabla\\cdot(\\mathbf{u}\\omega)`, de-aliased."""
        u, v = self.velocity(w_hat)
        w = np.fft.ifft2(w_hat).real
        uw_hat = np.fft.fft2(u * w)
        vw_hat = np.fft.fft2(v * w)
        rhs = -(1j * self.kx * uw_hat + 1j * self.ky * vw_hat)
        if self.dealias:
            rhs = rhs * self.dealias_mask
        return rhs

    def rhs(self, w_hat):
        """Complete right-hand side :math:`d\\hat{\\omega}/dt`."""
        return self.nonlinear_hat(w_hat) + self._viscous * w_hat

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def step_rk4(self, w_hat, dt):
        """Advance one RK4 step of size ``dt``; returns new ``w_hat``.

        The result is projected back onto the retained band so the state
        stays inside the alias-free Galerkin subspace.
        """
        k1 = self.rhs(w_hat)
        k2 = self.rhs(w_hat + 0.5 * dt * k1)
        k3 = self.rhs(w_hat + 0.5 * dt * k2)
        k4 = self.rhs(w_hat + dt * k3)
        result = w_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if self.dealias:
            result = result * self.dealias_mask
        return result

    def integrate(self, w0, dt, nsteps, sample_every=0):
        """Integrate a physical vorticity field; return states and times.

        The initial field is projected onto the retained band. A positive
        ``sample_every`` records every ``sample_every`` steps (always
        including step 0 and the final step); ``sample_every=0`` records the
        initial and final states only. The returned history is a list of
        ``(t, w_hat)`` tuples.
        """
        w_hat = np.fft.fft2(w0)
        if self.dealias:
            w_hat = w_hat * self.dealias_mask
        history = [(0.0, w_hat.copy())]
        for step in range(1, nsteps + 1):
            w_hat = self.step_rk4(w_hat, dt)
            if sample_every and (step % sample_every == 0 or step == nsteps):
                history.append((step * dt, w_hat.copy()))
        if not sample_every:
            history.append((nsteps * dt, w_hat.copy()))
        return history

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def energy(self, w_hat):
        """Mean kinetic energy per unit mass, ``0.5 <u^2+v^2>``."""
        u, v = self.velocity(w_hat)
        return 0.5 * float(np.mean(u * u + v * v))

    def enstrophy(self, w_hat):
        """Mean enstrophy, ``0.5 <omega^2>``."""
        w = np.fft.ifft2(w_hat).real
        return 0.5 * float(np.mean(w * w))

    def velocity_l2_error(self, w_hat, t):
        """L2 error of ``(u,v)`` against the Taylor-Green exact solution."""
        x = np.arange(self.n) * (2.0 * np.pi / self.n)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        u, v = self.velocity(w_hat)
        ue, ve = taylor_green_velocity(xx, yy, t, self.nu)
        err2 = float(np.mean((u - ue) ** 2 + (v - ve) ** 2))
        scale = float(np.mean(ue**2 + ve**2))
        return np.sqrt(err2 / scale)
