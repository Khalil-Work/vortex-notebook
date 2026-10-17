"""spectral_flow: periodic 2D incompressible Navier-Stokes solver.

Vorticity-streamfunction pseudospectral method with FFT operators,
de-aliased conservative nonlinear terms, and explicit RK4 time stepping.
"""

from .solver import SpectralVorticitySolver, taylor_green_vorticity

__all__ = ["SpectralVorticitySolver", "taylor_green_vorticity"]
__version__ = "1.0.0"
