# Model and method

## Problem

Two-dimensional incompressible Navier–Stokes equations on the periodic square
domain Ω = [0, 2π)², in vorticity–streamfunction form:

- ∂ω/∂t + ∇·(u ω) = ν ∇²ω   (vorticity transport)
- ω = ∂v/∂x − ∂u/∂y         (vorticity definition)
- u = ∂ψ/∂y, v = −∂ψ/∂x     (incompressible velocity from a streamfunction)

Symbols: u = (u, v) velocity [m/s]; ω vorticity [1/s]; ψ streamfunction
[m²/s]; ν kinematic viscosity [m²/s]; t time [s]. Combining the last two
relations gives the Poisson equation for the streamfunction,

    ω = −∇²ψ.

## Spatial discretization

Everything happens in Fourier space on an N×N uniform grid. For a periodic
mode e^{i k·x}, ∇ → i k, so

- Poisson inversion:  ψ̂ = ω̂ / k²  for k ≠ 0, and ψ̂(0,0) = 0.
- Velocity recovery:  û = i k_y ψ̂,  v̂ = −i k_x ψ̂.
- Viscous term:       ν∇²ω → −ν k² ω̂.

The streamfunction zero mode only fixes the arbitrary additive constant of ψ
(it carries no velocity).

### Nonlinear term and de-aliasing

The advective term is written in conservative form −∇·(u ω): the products
u ω and v ω are formed pointwise in physical space, transformed, and
differentiated spectrally. Quadratic products of band-limited fields are not
band-limited, and discrete FFTs fold their high modes back onto resolved
ones (aliasing).

The implemented policy keeps the band |k_x| < N/3 and |k_y| < N/3
**strictly**. Both the solution and the nonlinear term are projected onto
that band (input and output truncation). The strict inequality matters: with
an inclusive cutoff at exactly N/3 (possible when N is divisible by 3), two
boundary modes at N/3 sum to 2N/3, which folds to 2N/3 − N = −N/3 — inside
the band — reintroducing aliasing error. With the strict band, any folded
sum 2k − N of retained modes lies below −N/3, outside the band, so the
discrete quadratic nonlinearity is **alias-free on the retained band**, and
the scheme is an exact Galerkin integration of Navier–Stokes restricted to
that band. The truncation of true product content between N/3 and 2N/3 is
the standard, deliberate Galerkin closure of the 2/3 rule.

### Mean vorticity

The k = 0 coefficient of the right-hand side is identically zero (the
spectral divergence of any field has zero mean; the viscous term vanishes at
k = 0), so ω̂(0,0) is an exactly conserved passive constant. A periodic
single-valued streamfunction cannot represent a uniform background curl, so
the recovered velocity is the curl-inverse of the zero-mean part of ω; the
curl round trip ω → ψ → u → curl returns ω with its mean removed. For the 2D
periodic flows targeted here, the mean does not participate in the dynamics.

## Time integration

Classical explicit RK4 with constant Δt. Stability limits (rule of thumb for
this operator):

- diffusion: ν |k|²_max Δt ≲ 2.8, |k|²_max = 2 (N/2)²;
- convection: max|u| Δt / Δx ≲ 1.

Taylor–Green is a single resolved eigenmode of the complete discrete
operator, so its RK4 integration error behaves like the scalar
e^{λΔt} expansion and decays as Δt⁴.

## Diagnostics

- Kinetic energy E = ⟨u² + v²⟩/2 [m²/s²] and enstrophy Z = ⟨ω²⟩/2 [1/s²],
  evaluated from the recovered fields.
- Shell-averaged energy spectrum E(|k|): energy density binned by |k| = √(k_x² + k_y²).

## Analytical reference

The Taylor–Green vortex (Taylor 1937; see e.g. Batchelor, *An Introduction to
Fluid Dynamics*):

    u(x,y,t) = −cos x sin y f(t),   v(x,y,t) = sin x cos y f(t),
    ω(x,y,t) = 2 cos x cos y f(t),  f(t) = exp(−2 ν t),

    E(t) = ¼ e^{−4νt},   Z(t) = ½ e^{−4νt}.

For this field the nonlinear term vanishes identically, which makes it an
exact solution of the full equations and a clean verification of the
viscous operator, the Poisson inversion, and the time integrator (the
nonlinear machinery is verified separately against a padded reference; see
verification.md).
