# Verification evidence

All claims below are reproduced by `python -m pytest tests -q` (12 tests) and
`python examples/run_verification.py` (figures + raw CSV/JSON under
`docs/results/`). Verified on Windows 11, Python 3.13.5, NumPy 2.1.3.

## What each test establishes

| Test | Property established |
| --- | --- |
| `test_velocity_recovery_from_taylor_green_vorticity` | Poisson sign and velocity recovery: recovered (u,v) match the analytic Taylor–Green velocity to 1e-12. Catches the classic ψ̂ = −ω̂/k² sign error. |
| `test_velocity_is_divergence_free_to_machine_precision` | Spectral divergence i(k_x û + k_y v̂) ≡ 0 for a random field, including retained high modes. |
| `test_curl_round_trip_recovers_vorticity_at_all_retained_modes` | ω → ψ → (u,v) → curl returns ω to 1e-12 on every retained mode of a random band-limited field. |
| `test_nonlinear_operator_matches_padded_reference` | Behavioral alias check: the de-aliased nonlinear term equals an independent 3/2-rule zero-padded (72-point for N=48) evaluation, restricted and truncated identically, to 1e-12 relative on random retained-band input. |
| `test_dealias_mask_structure` | Retained band is \|k\| < N/3 strictly: on N=48 mode 15 is kept, boundary mode 16 is excluded (16+16=32 folds onto retained −16 if included). |
| `test_taylor_green_decay_accuracy` | Full-solver accuracy: relative L2 velocity error at t = 1 s (N=128, ν=0.01, Δt=0.002) < 1e-7; measured 3.9e-16 (roundoff floor). |
| `test_fourth_order_time_convergence` | RK4 order: N=8, ν=0.5, T=1, Δt ∈ {0.2, 0.1, 0.05, 0.025}; measured pair orders 4.12 → 4.03 (> 3.5 asserted). Δt=0.025 keeps error ≈1e-9, far above the ~1e-15 roundoff floor, so no saturation contaminates the slope. |
| `test_mean_vorticity_is_passive_and_velocity_curl_has_zero_mean` | ω̂(0,0) is exactly conserved; recovered-velocity curl has zero mean (documented periodic-curl limitation). |
| `test_mean_vorticity_conserved_under_nonlinear_dynamics` | Zero mode stays zero under nonlinear dynamics with a nonzero-mean-free IC. |
| `test_nonlinear_operator_matches_finite_differences` | The nonlinear operator agrees with an independent central finite-difference evaluation on a smooth field to within FD truncation (~1.5% at this grid). |
| `test_energy_and_enstrophy_decay_exactly_for_taylor_green` | E(t) = ¼e^{−4νt} and Z(t) = ½e^{−4νt} to 1e-8 relative at t = 1 s. |
| `test_invalid_constructor_arguments` | Input validation. |

## What the recorded run shows (`docs/results/`)

- `taylor_green_history.csv` + `taylor_green_verification.png`: N=128,
  ν=0.01, Δt=0.001, t ∈ [0, 2]. Energy and enstrophy track the analytic
  decay curves; the relative L2 velocity error stays at the ~1e-15 roundoff
  floor (spectral spatial exactness for a band-limited solution; the RK4
  truncation error at this Δt is far below roundoff).
- `time_convergence.csv`: strongly damped regime (N=8, ν=0.5, T=1) where
  the temporal error is resolvable; observed orders 4.12, 4.06, 4.03 with
  slope-4 reference line in the figure.
- `nonlinear_cascade.png/.svg`, `nonlinear_history.csv`,
  `nonlinear_spectra.csv`: computed demonstration (N=128, ν=5e-4, t=8 s)
  with band-limited large-scale IC plus small filtered noise (seed
  20261018). Vortex merger, enstrophy decay, and spectral broadening are
  outputs of the computed run; there is **no external reference** for this
  case and none is claimed. Energy is nearly conserved on this interval
  because ν is small.
- `run_summary.json`: machine/runtime context and the metrics above.

## Limits of this evidence

- Taylor–Green verifies the linear machinery exactly (its nonlinear term is
  identically zero); nonlinear correctness is established by the padded
  3/2-rule reference, not by an external turbulent-flow benchmark.
- No comparison against experimental data, no turbulence-statistics
  validation (e.g. enstrophy production budgets), and no high-Re study.
- The nonlinear demonstration is illustrative: conclusions about cascade
  physics would require resolution and Re sweeps.
