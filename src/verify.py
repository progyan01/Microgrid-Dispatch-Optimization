"""
Verification checks for solved easy and hard model results (§7 of the project spec).

Functions
    summarise_easy(res)   — LP sanity checks
    summarise_hard(res)   — MILP sanity checks (superset of easy checks)
    verify_both(easy, hard) — side-by-side comparison and price-of-realism header

All checks print to stdout; nothing is returned.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DT, E_CAP, P_BAR, P_MIN, ETA_C, ETA_D,
    E_MIN, E_MAX, E_INIT, LAMBDA,
    D_OFF, D_ON, C_DEG, R_BAT, R_GRID,
)
from easy_model import EasyResult, load as load_easy
from hard_model import HardResult, load as load_hard

TOL = 1e-4   # tolerance for near-zero checks


# shared helper

def _battery_dynamics_error(res) -> float:
    """Max deviation between solver E array and E recomputed from pc/pd."""
    T = len(res.df)
    E_check = np.empty(T + 1)
    E_check[0] = res.E[0]
    for t in range(T):
        E_check[t + 1] = (1 - LAMBDA * DT) * E_check[t] + ETA_C * res.pc[t] * DT - res.pd[t] * DT / ETA_D
    return float(np.abs(E_check - res.E).max())


def _demand_recompute(res):
    """Recompute monthly peak draws from solution arrays. Returns (pk_off_12, pk_on_12)."""
    month = res.df["month"].values
    on    = res.df["period"].values == 2
    pk_off = np.array([res.g[(month == m) & ~on].max() for m in range(1, 13)])
    pk_on  = np.array([res.g[(month == m) &  on].max() for m in range(1, 13)])
    return pk_off, pk_on


def _throughput(res) -> float:
    return float((res.pc + res.pd).sum() * DT)


# easy model

def summarise_easy(res: EasyResult) -> None:
    """Print LP solution summary and all §7 checks for the easy model."""
    df    = res.df
    T     = len(df)
    L     = df["L"].values
    S     = df["S"].values

    print("=" * 55)
    print("Easy model (LP) — verification")
    print("=" * 55)
    print(f"  Solver status   : {res.status}")
    print(f"  Total cost      : ${res.cost:,.2f}")
    print(f"    Energy bill   : ${res.energy_bill:,.2f}")
    print(f"    Demand charge : ${res.demand_bill:,.2f}")
    print()

    # Power balance
    residual = np.abs(res.g + res.s + res.pd - L - res.pc)
    print(f"  Power balance — max residual        : {residual.max():.2e} kW")

    # Battery energy bounds and cyclic boundary
    print(f"  Battery E — min / max               : {res.E.min():.2f} / {res.E.max():.2f} kWh"
          f"  (limits {E_MIN:.1f} / {E_MAX:.1f})")
    print(f"  E[0] = {res.E[0]:.2f}, E[T] = {res.E[T]:.2f}  (target {E_INIT:.1f})")

    # Battery dynamics
    dyn_err = _battery_dynamics_error(res)
    print(f"  Battery dynamics recheck            : {dyn_err:.2e} kWh max error")

    # Power limits
    pc_over = np.sum(res.pc > P_BAR + TOL)
    pd_over = np.sum(res.pd > P_BAR + TOL)
    print(f"  Power limit violations (> P_BAR)    : pc={pc_over}, pd={pd_over}")

    # Minimum operating power (informational — not a constraint in the easy model)
    pc_below = np.sum((res.pc > TOL) & (res.pc < P_MIN - TOL))
    pd_below = np.sum((res.pd > TOL) & (res.pd < P_MIN - TOL))
    print(f"  Hours below P_MIN (easy has no limit): pc={pc_below}, pd={pd_below}")

    # Simultaneous charge+discharge
    simul = np.sum((res.pc > TOL) & (res.pd > TOL))
    print(f"  Hours with simultaneous charge+discharge : {simul}")

    # Solar use
    solar_over = np.sum(res.s > S + TOL)
    curtailed  = float((S - res.s).sum())
    print(f"  Solar used > available              : {solar_over} hours")
    print(f"  Solar curtailed                     : {curtailed:,.0f} kWh"
          f"  ({100*curtailed/S.sum():.1f}% of available)")

    # Demand charge cross-check
    pk_off_chk, pk_on_chk = _demand_recompute(res)
    demand_chk = D_OFF * pk_off_chk.sum() + D_ON * pk_on_chk.sum()
    print(f"  Demand charge recheck               : solver ${res.demand_bill:,.2f}"
          f" vs recomputed ${demand_chk:,.2f}")

    # Cost recomputation
    cost_chk = float(df["c"].values @ res.g * DT) + demand_chk
    print(f"  Cost recheck                        : solver ${res.cost:,.2f}"
          f" vs recomputed ${cost_chk:,.2f}")

    # Throughput
    tp = _throughput(res)
    print(f"  Annual throughput                   : {tp:,.0f} kWh"
          f"  ({tp / (2 * E_CAP):.0f} equivalent full cycles)")
    print("=" * 55)


# hard model

def summarise_hard(res: HardResult) -> None:
    """Print MILP solution summary and all §7 checks for the hard model."""
    df    = res.df
    T     = len(df)
    L     = df["L"].values
    S     = df["S"].values

    print("=" * 55)
    print("Hard model (MILP) — verification")
    print("=" * 55)
    print(f"  Solver status   : {res.status}")
    if not np.isnan(res.mip_gap):
        print(f"  MIP gap         : {res.mip_gap:.4f}")
    print(f"  Total cost      : ${res.cost:,.2f}")
    print(f"    Energy bill   : ${res.energy_bill:,.2f}")
    print(f"    Demand charge : ${res.demand_bill:,.2f}")
    print(f"    Wear cost     : ${res.wear_bill:,.2f}")
    print()

    # Power balance
    residual = np.abs(res.g + res.s + res.pd - L - res.pc)
    print(f"  Power balance — max residual        : {residual.max():.2e} kW")

    # Battery energy bounds and cyclic boundary
    print(f"  Battery E — min / max               : {res.E.min():.2f} / {res.E.max():.2f} kWh"
          f"  (limits {E_MIN:.1f} / {E_MAX:.1f})")
    print(f"  E[0] = {res.E[0]:.2f}, E[T] = {res.E[T]:.2f}  (target {E_INIT:.1f})")

    # Battery dynamics
    dyn_err = _battery_dynamics_error(res)
    print(f"  Battery dynamics recheck            : {dyn_err:.2e} kWh max error")

    # Minimum operating power
    pc_viol = np.sum((res.pc > TOL) & (res.pc < P_MIN - TOL))
    pd_viol = np.sum((res.pd > TOL) & (res.pd < P_MIN - TOL))
    print(f"  P_MIN violations                    : pc={pc_viol}, pd={pd_viol}")

    # No simultaneous charge+discharge
    simul = np.sum((res.pc > TOL) & (res.pd > TOL))
    print(f"  Hours with simultaneous charge+discharge : {simul}")

    # Binary consistency: active power only when binary = 1
    uc_mismatch = np.sum((res.pc > TOL) & (res.uc < 0.5))
    ud_mismatch = np.sum((res.pd > TOL) & (res.ud < 0.5))
    print(f"  Binary mismatches                   : uc={uc_mismatch}, ud={ud_mismatch}")

    # Battery ramp
    net = res.pd - res.pc
    ramp_viol = np.sum(np.abs(np.diff(net)) > R_BAT + TOL)
    print(f"  Battery ramp violations (>{R_BAT} kW)  : {ramp_viol}")

    # Grid ramp
    if R_GRID is not None:
        gramp_viol = np.sum(np.abs(np.diff(res.g)) > R_GRID + TOL)
        print(f"  Grid ramp violations (>{R_GRID} kW)    : {gramp_viol}")

    # Solar use
    solar_over = np.sum(res.s > S + TOL)
    curtailed  = float((S - res.s).sum())
    print(f"  Solar used > available              : {solar_over} hours")
    print(f"  Solar curtailed                     : {curtailed:,.0f} kWh"
          f"  ({100*curtailed/S.sum():.1f}% of available)")

    # Demand charge cross-check
    pk_off_chk, pk_on_chk = _demand_recompute(res)
    demand_chk = D_OFF * pk_off_chk.sum() + D_ON * pk_on_chk.sum()
    print(f"  Demand charge recheck               : solver ${res.demand_bill:,.2f}"
          f" vs recomputed ${demand_chk:,.2f}")

    # Cost recomputation
    wear_chk = C_DEG * (res.pc + res.pd).sum() * DT
    cost_chk = float(df["c"].values @ res.g * DT) + demand_chk + wear_chk
    print(f"  Cost recheck                        : solver ${res.cost:,.2f}"
          f" vs recomputed ${cost_chk:,.2f}")

    # Mode switches per day
    mode     = np.where(res.uc > 0.5, 1, np.where(res.ud > 0.5, -1, 0))
    switches = int(np.sum(mode[1:] != mode[:-1]))
    print(f"  Mode switches per day (avg)         : {switches / 365:.1f}")

    # Throughput
    tp = _throughput(res)
    print(f"  Annual throughput                   : {tp:,.0f} kWh"
          f"  ({tp / (2 * E_CAP):.0f} equivalent full cycles)")
    print("=" * 55)


# side-by-side

def verify_both(easy: EasyResult, hard: HardResult) -> None:
    """Run verification for both models, then print the price-of-realism summary."""
    summarise_easy(easy)
    print()
    summarise_hard(hard)
    print()

    price_of_realism = hard.cost - easy.cost
    pct              = 100 * price_of_realism / easy.cost

    # Split: wear paid vs extra energy+demand from constraints
    extra_ops = (hard.energy_bill + hard.demand_bill) - (easy.energy_bill + easy.demand_bill)

    print("=" * 55)
    print("Price of realism")
    print("=" * 55)
    print(f"  Easy cost       : ${easy.cost:,.2f}")
    print(f"  Hard cost       : ${hard.cost:,.2f}")
    print(f"  Difference      : ${price_of_realism:,.2f}  ({pct:.2f}% of easy cost)")
    print(f"    of which wear : ${hard.wear_bill:,.2f}")
    print(f"    of which extra energy+demand from constraints : ${extra_ops:,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    easy = load_easy()
    hard = load_hard()
    verify_both(easy, hard)
