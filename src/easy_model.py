"""
Solves the easy (LP) economic dispatch problem for the full year.

Decision variables (T = 8760 hourly steps)
    g[t]        grid import (kW)
    s[t]        solar power used (kW); surplus is curtailed
    pc[t]       battery charge power (kW)
    pd[t]       battery discharge power (kW)
    E[t]        battery stored energy at start of hour t (kWh); length T+1
    pk_off[m]   monthly peak grid draw during off-peak hours (kW); m = 0..11
    pk_on[m]    monthly peak grid draw during on-peak hours (kW);  m = 0..11

Objective
    minimize  energy bill  +  monthly demand charges

No wear cost, no minimum power, no ramp limits, no binary variables — pure LP.
Solved with scipy.optimize.linprog (HiGHS backend).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog, LinearConstraint, Bounds
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DT, E_CAP, P_BAR, ETA_C, ETA_D,
    E_MIN, E_MAX, E_INIT, LAMBDA,
    D_OFF, D_ON, RESULTS_DIR,
)
from data_prep import build_dataset


@dataclass
class EasyResult:
    """All outputs from a solved easy-model run."""
    status:       str            # solver status string
    cost:         float          # optimal total cost ($)
    energy_bill:  float          # energy component of cost ($)
    demand_bill:  float          # demand-charge component of cost ($)
    g:            np.ndarray     # grid import (kW), length T
    s:            np.ndarray     # solar used (kW), length T
    pc:           np.ndarray     # battery charge power (kW), length T
    pd:           np.ndarray     # battery discharge power (kW), length T
    E:            np.ndarray     # battery energy (kWh), length T+1
    pk_off:       np.ndarray     # monthly off-peak peak draw (kW), length 12
    pk_on:        np.ndarray     # monthly on-peak peak draw (kW), length 12
    df:           pd.DataFrame   # original input table (L, S, c, period, month)


def _build_lp(df: pd.DataFrame):
    """
    Assemble the LP in the form expected by scipy.optimize.linprog.

    Variable layout (all concatenated into one vector x):
        x[0  : T]       g      (T vars)
        x[T  : 2T]      s      (T vars)
        x[2T : 3T]      pc     (T vars)
        x[3T : 4T]      pd     (T vars)
        x[4T : 5T+1]    E      (T+1 vars)
        x[5T+1: 5T+13]  pk_off (12 vars)
        x[5T+13: 5T+25] pk_on  (12 vars)

    Total: 5T + 25 variables.
    """
    T      = len(df)
    L      = df["L"].values       # load (kW)
    S      = df["S"].values       # available PV (kW)
    c      = df["c"].values       # energy price ($/kWh)
    month  = df["month"].values   # 1-indexed calendar month
    on     = df["period"].values == 2   # True for on-peak hours

    N = 5 * T + 25   # total number of decision variables

    # Offsets into x
    i_g      = 0
    i_s      = T
    i_pc     = 2 * T
    i_pd     = 3 * T
    i_E      = 4 * T          # length T+1
    i_pkoff  = 5 * T + 1      # length 12
    i_pkon   = 5 * T + 13     # length 12

    # Objective: c^T g * dt + D_OFF * sum(pk_off) + D_ON * sum(pk_on)
    obj = np.zeros(N)
    obj[i_g : i_g + T]       = c * DT
    obj[i_pkoff : i_pkoff + 12] = D_OFF
    obj[i_pkon  : i_pkon  + 12] = D_ON

    # Variable bounds
    lb = np.zeros(N)
    ub = np.full(N, np.inf)

    ub[i_g  : i_g  + T] = np.inf      # grid import unbounded above (demand charge handles it)
    ub[i_s  : i_s  + T] = S           # solar used <= available solar
    ub[i_pc : i_pc + T] = P_BAR       # charge power <= P_BAR
    ub[i_pd : i_pd + T] = P_BAR       # discharge power <= P_BAR
    lb[i_E  : i_E  + T + 1] = E_MIN  # stored energy >= E_MIN
    ub[i_E  : i_E  + T + 1] = E_MAX  # stored energy <= E_MAX
    # pk_off, pk_on already have lb=0, ub=inf

    # Equality constraints
    # 1. Power balance:  g + s + pd - pc = L   (T equations)
    # 2. Battery dynamics: E[t+1] - (1-lam)*E[t] - eta_c*pc[t] + pd[t]/eta_d = 0  (T equations)
    # 3. Cyclic boundary: E[0] = E_INIT, E[T] = E_INIT  (2 equations)
    n_eq = T + T + 2
    A_eq = np.zeros((n_eq, N))
    b_eq = np.zeros(n_eq)

    rows = np.arange(T)

    # Power balance rows 0..T-1
    A_eq[rows, i_g  + rows] =  1.0
    A_eq[rows, i_s  + rows] =  1.0
    A_eq[rows, i_pd + rows] =  1.0
    A_eq[rows, i_pc + rows] = -1.0
    b_eq[rows] = L

    # Battery dynamics rows T..2T-1
    r = T + rows
    A_eq[r, i_E + rows + 1] =  1.0                        # E[t+1]
    A_eq[r, i_E + rows]     = -(1.0 - LAMBDA * DT)        # -(1 - λ·dt)·E[t]
    A_eq[r, i_pc + rows]    = -ETA_C * DT                 # -η_c·pc[t]·dt
    A_eq[r, i_pd + rows]    =  DT / ETA_D                 # +pd[t]·dt/η_d
    b_eq[r] = 0.0

    # Cyclic boundary: E[0] = E_INIT
    A_eq[2 * T,     i_E]       = 1.0
    b_eq[2 * T]                = E_INIT
    # Cyclic boundary: E[T] = E_INIT
    A_eq[2 * T + 1, i_E + T]  = 1.0
    b_eq[2 * T + 1]            = E_INIT

    # Inequality constraints: g[t] <= pk_off[m] or pk_on[m] for each hour
    # Written as:  g[t] - pk_off[m(t)] <= 0  (off-peak hours)
    #              g[t] - pk_on[m(t)]  <= 0  (on-peak hours)
    n_ineq = T
    A_ub = np.zeros((n_ineq, N))
    b_ub = np.zeros(n_ineq)

    for t in range(T):
        m_idx = int(month[t]) - 1   # 0-indexed month
        A_ub[t, i_g + t] = 1.0
        if on[t]:
            A_ub[t, i_pkon  + m_idx] = -1.0
        else:
            A_ub[t, i_pkoff + m_idx] = -1.0

    bounds = Bounds(lb=lb, ub=ub)

    return obj, A_eq, b_eq, A_ub, b_ub, bounds, {
        "T": T, "i_g": i_g, "i_s": i_s, "i_pc": i_pc,
        "i_pd": i_pd, "i_E": i_E, "i_pkoff": i_pkoff, "i_pkon": i_pkon,
    }


def solve(df: pd.DataFrame | None = None) -> EasyResult:
    """
    Build and solve the easy LP. If df is None, calls build_dataset() automatically.
    Returns an EasyResult with all decision variable arrays and cost components.
    """
    if df is None:
        df = build_dataset()

    obj, A_eq, b_eq, A_ub, b_ub, bounds, idx = _build_lp(df)

    result = linprog(
        c       = obj,
        A_ub    = A_ub,
        b_ub    = b_ub,
        A_eq    = A_eq,
        b_eq    = b_eq,
        bounds  = list(zip(bounds.lb, bounds.ub)),   # linprog expects [(lb, ub), ...]
        method  = "highs",
        options = {"disp": False},
    )

    if result.status != 0:
        raise RuntimeError(f"LP did not solve to optimality — status {result.status}: {result.message}")

    T      = idx["T"]
    x      = result.x
    g      = x[idx["i_g"]   : idx["i_g"]  + T]
    s      = x[idx["i_s"]   : idx["i_s"]  + T]
    pc     = x[idx["i_pc"]  : idx["i_pc"] + T]
    pd_    = x[idx["i_pd"]  : idx["i_pd"] + T]
    E      = x[idx["i_E"]   : idx["i_E"]  + T + 1]
    pk_off = x[idx["i_pkoff"] : idx["i_pkoff"] + 12]
    pk_on  = x[idx["i_pkon"]  : idx["i_pkon"]  + 12]

    energy_bill = float(df["c"].values @ g * DT)
    demand_bill = float(D_OFF * pk_off.sum() + D_ON * pk_on.sum())

    return EasyResult(
        status      = result.message,
        cost        = float(result.fun),
        energy_bill = energy_bill,
        demand_bill = demand_bill,
        g=g, s=s, pc=pc, pd=pd_, E=E,
        pk_off=pk_off, pk_on=pk_on,
        df=df,
    )


def save(res: EasyResult, path: Path | None = None) -> Path:
    """Pickle the EasyResult to disk. Returns the path written."""
    out = Path(path) if path else RESULTS_DIR / "easy_result.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"Easy result saved → {out}")
    return out


def load(path: Path | None = None) -> EasyResult:
    """Load a previously saved EasyResult."""
    src = Path(path) if path else RESULTS_DIR / "easy_result.pkl"

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # pickle saved under __main__ when run as a script; remap to this module
            if module == "__main__":
                module = "easy_model"
            return super().find_class(module, name)

    with open(src, "rb") as f:
        return _Unpickler(f).load()


if __name__ == "__main__":
    print("Building dataset...")
    df = build_dataset()
    print(f"Solving LP over {len(df)} hours...")
    res = solve(df)
    save(res)
