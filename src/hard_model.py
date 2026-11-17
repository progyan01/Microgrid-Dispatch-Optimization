"""
Solves the hard (MILP) economic dispatch problem for the full year.

Decision variables (T = 8760 hourly steps)
    g[t]        grid import (kW)
    s[t]        solar power used (kW); surplus is curtailed
    pc[t]       battery charge power (kW)
    pd[t]       battery discharge power (kW)
    E[t]        battery stored energy at start of hour t (kWh); length T+1
    pk_off[m]   monthly peak grid draw during off-peak hours (kW); m = 0..11
    pk_on[m]    monthly peak grid draw during on-peak hours (kW);  m = 0..11
    uc[t]       1 if battery is charging in hour t, else 0  (binary)
    ud[t]       1 if battery is discharging in hour t, else 0  (binary)

Objective
    minimize  energy bill  +  monthly demand charges  +  battery wear cost

Adds over the easy model: wear cost (C_DEG per kWh throughput), minimum
operating power, no simultaneous charge+discharge, and battery ramp limits.
Solved with scipy.optimize.milp (HiGHS backend).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, vstack as sp_vstack
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DT, E_CAP, P_BAR, P_MIN, ETA_C, ETA_D,
    E_MIN, E_MAX, E_INIT, LAMBDA,
    D_OFF, D_ON, C_DEG, R_BAT, R_GRID, RESULTS_DIR,
)
from data_prep import build_dataset


@dataclass
class HardResult:
    """All outputs from a solved hard-model run."""
    status:       str          # solver status string
    mip_gap:      float        # relative optimality gap reported by HiGHS
    cost:         float        # optimal total cost ($)
    energy_bill:  float        # energy component of cost ($)
    demand_bill:  float        # demand-charge component of cost ($)
    wear_bill:    float        # battery wear component of cost ($)
    g:            np.ndarray   # grid import (kW), length T
    s:            np.ndarray   # solar used (kW), length T
    pc:           np.ndarray   # battery charge power (kW), length T
    pd:           np.ndarray   # battery discharge power (kW), length T
    E:            np.ndarray   # battery energy (kWh), length T+1
    uc:           np.ndarray   # charge binary, length T
    ud:           np.ndarray   # discharge binary, length T
    pk_off:       np.ndarray   # monthly off-peak peak draw (kW), length 12
    pk_on:        np.ndarray   # monthly on-peak peak draw (kW), length 12
    df:           pd.DataFrame # original input table (L, S, c, period, month)


def _build_milp(df: pd.DataFrame):
    """
    Assemble the MILP in the form expected by scipy.optimize.milp.

    Variable layout (all concatenated into one vector x):
        x[0    : T]       g       (T continuous)
        x[T    : 2T]      s       (T continuous)
        x[2T   : 3T]      pc      (T continuous)
        x[3T   : 4T]      pd      (T continuous)
        x[4T   : 5T+1]    E       (T+1 continuous)
        x[5T+1 : 5T+13]   pk_off  (12 continuous)
        x[5T+13: 5T+25]   pk_on   (12 continuous)
        x[5T+25: 6T+25]   uc      (T binary)
        x[6T+25: 7T+25]   ud      (T binary)

    Total: 7T + 25 variables.
    """
    T     = len(df)
    L     = df["L"].values
    S     = df["S"].values
    c     = df["c"].values
    month = df["month"].values
    on    = df["period"].values == 2

    N = 7 * T + 25

    # Variable index offsets
    i_g     = 0
    i_s     = T
    i_pc    = 2 * T
    i_pd    = 3 * T
    i_E     = 4 * T          # length T+1
    i_pkoff = 5 * T + 1      # length 12
    i_pkon  = 5 * T + 13     # length 12
    i_uc    = 5 * T + 25     # length T  (binary)
    i_ud    = 6 * T + 25     # length T  (binary)

    # Objective: energy bill + demand charges + wear cost on throughput
    obj = np.zeros(N)
    obj[i_g    : i_g    + T]   = c * DT
    obj[i_pkoff: i_pkoff + 12] = D_OFF
    obj[i_pkon : i_pkon  + 12] = D_ON
    obj[i_pc   : i_pc   + T]   = C_DEG * DT   # wear on charge kWh
    obj[i_pd   : i_pd   + T]   = C_DEG * DT   # wear on discharge kWh

    # Variable bounds
    lb = np.zeros(N)
    ub = np.full(N, np.inf)

    ub[i_s  : i_s  + T]     = S       # solar used <= available solar
    ub[i_pc : i_pc + T]     = P_BAR   # charge power <= P_BAR
    ub[i_pd : i_pd + T]     = P_BAR   # discharge power <= P_BAR
    lb[i_E  : i_E  + T + 1] = E_MIN
    ub[i_E  : i_E  + T + 1] = E_MAX
    ub[i_uc : i_uc + T]     = 1.0     # binary uc in {0, 1}
    ub[i_ud : i_ud + T]     = 1.0     # binary ud in {0, 1}

    bounds = Bounds(lb=lb, ub=ub)

    # Integer variable indices: uc and ud are binary
    integrality = np.zeros(N)
    integrality[i_uc: i_uc + T] = 1
    integrality[i_ud: i_ud + T] = 1

    rows = np.arange(T)

    # --- Equality constraints ---
    # 1. Power balance:  g + s + pd - pc = L                  (T rows)
    # 2. Battery dynamics: E[t+1] - (1-lam)*E[t] - eta_c*pc + pd/eta_d = 0  (T rows)
    # 3. Cyclic boundary: E[0] = E_INIT, E[T] = E_INIT        (2 rows)

    n_eq = 2 * T + 2
    A_eq = lil_matrix((n_eq, N))
    b_eq_lo = np.zeros(n_eq)
    b_eq_hi = np.zeros(n_eq)

    # Power balance
    A_eq[rows, i_g  + rows] =  1.0
    A_eq[rows, i_s  + rows] =  1.0
    A_eq[rows, i_pd + rows] =  1.0
    A_eq[rows, i_pc + rows] = -1.0
    b_eq_lo[rows] = L
    b_eq_hi[rows] = L

    # Battery dynamics
    r = T + rows
    A_eq[r, i_E + rows + 1] =  1.0
    A_eq[r, i_E + rows]     = -(1.0 - LAMBDA * DT)
    A_eq[r, i_pc + rows]    = -ETA_C * DT
    A_eq[r, i_pd + rows]    =  DT / ETA_D
    # b = 0 already

    # Cyclic boundary
    A_eq[2 * T,     i_E]     = 1.0
    b_eq_lo[2 * T]            = E_INIT
    b_eq_hi[2 * T]            = E_INIT
    A_eq[2 * T + 1, i_E + T] = 1.0
    b_eq_lo[2 * T + 1]        = E_INIT
    b_eq_hi[2 * T + 1]        = E_INIT

    # --- Inequality constraints ---
    # Collected as (lo <= A x <= hi); use -inf for one-sided.

    ineq_rows = []
    ineq_lo   = []
    ineq_hi   = []

    # Demand charge peak bounds: g[t] <= pk_off[m] or pk_on[m]
    #   g[t] - pk_off[m] <= 0  for off-peak hours
    #   g[t] - pk_on[m]  <= 0  for on-peak hours
    A_pk = lil_matrix((T, N))
    for t in range(T):
        m_idx = int(month[t]) - 1
        A_pk[t, i_g + t] = 1.0
        if on[t]:
            A_pk[t, i_pkon  + m_idx] = -1.0
        else:
            A_pk[t, i_pkoff + m_idx] = -1.0
    ineq_rows.append(A_pk)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.zeros(T))

    # No simultaneous charge+discharge: uc + ud <= 1
    A_simul = lil_matrix((T, N))
    A_simul[rows, i_uc + rows] = 1.0
    A_simul[rows, i_ud + rows] = 1.0
    ineq_rows.append(A_simul)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.ones(T))

    # Charge: P_MIN * uc <= pc <= P_BAR * uc
    #   pc - P_BAR * uc <= 0   (upper bound links pc to uc)
    #   -pc + P_MIN * uc <= 0  (lower bound: if uc=1 then pc >= P_MIN)
    A_pc_up = lil_matrix((T, N))
    A_pc_up[rows, i_pc + rows] =  1.0
    A_pc_up[rows, i_uc + rows] = -P_BAR
    ineq_rows.append(A_pc_up)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.zeros(T))

    A_pc_lo = lil_matrix((T, N))
    A_pc_lo[rows, i_pc + rows] = -1.0
    A_pc_lo[rows, i_uc + rows] =  P_MIN
    ineq_rows.append(A_pc_lo)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.zeros(T))

    # Discharge: P_MIN * ud <= pd <= P_BAR * ud
    A_pd_up = lil_matrix((T, N))
    A_pd_up[rows, i_pd + rows] =  1.0
    A_pd_up[rows, i_ud + rows] = -P_BAR
    ineq_rows.append(A_pd_up)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.zeros(T))

    A_pd_lo = lil_matrix((T, N))
    A_pd_lo[rows, i_pd + rows] = -1.0
    A_pd_lo[rows, i_ud + rows] =  P_MIN
    ineq_rows.append(A_pd_lo)
    ineq_lo.append(np.full(T, -np.inf))
    ineq_hi.append(np.zeros(T))

    # Battery ramp: |(pd[t] - pc[t]) - (pd[t-1] - pc[t-1])| <= R_BAT
    #   net[t] - net[t-1] <=  R_BAT
    #   net[t-1] - net[t] <=  R_BAT
    t_rng  = rows[1:]         # t = 1..T-1
    n_ramp = T - 1

    A_ramp_up = lil_matrix((n_ramp, N))
    A_ramp_up[np.arange(n_ramp), i_pd + t_rng]     =  1.0
    A_ramp_up[np.arange(n_ramp), i_pc + t_rng]     = -1.0
    A_ramp_up[np.arange(n_ramp), i_pd + t_rng - 1] = -1.0
    A_ramp_up[np.arange(n_ramp), i_pc + t_rng - 1] =  1.0
    ineq_rows.append(A_ramp_up)
    ineq_lo.append(np.full(n_ramp, -np.inf))
    ineq_hi.append(np.full(n_ramp, R_BAT))

    A_ramp_dn = lil_matrix((n_ramp, N))
    A_ramp_dn[np.arange(n_ramp), i_pd + t_rng]     = -1.0
    A_ramp_dn[np.arange(n_ramp), i_pc + t_rng]     =  1.0
    A_ramp_dn[np.arange(n_ramp), i_pd + t_rng - 1] =  1.0
    A_ramp_dn[np.arange(n_ramp), i_pc + t_rng - 1] = -1.0
    ineq_rows.append(A_ramp_dn)
    ineq_lo.append(np.full(n_ramp, -np.inf))
    ineq_hi.append(np.full(n_ramp, R_BAT))

    # Grid ramp: |g[t] - g[t-1]| <= R_GRID  (only if R_GRID is set)
    if R_GRID is not None:
        A_gramp_up = lil_matrix((n_ramp, N))
        A_gramp_up[np.arange(n_ramp), i_g + t_rng]     =  1.0
        A_gramp_up[np.arange(n_ramp), i_g + t_rng - 1] = -1.0
        ineq_rows.append(A_gramp_up)
        ineq_lo.append(np.full(n_ramp, -np.inf))
        ineq_hi.append(np.full(n_ramp, R_GRID))

        A_gramp_dn = lil_matrix((n_ramp, N))
        A_gramp_dn[np.arange(n_ramp), i_g + t_rng]     = -1.0
        A_gramp_dn[np.arange(n_ramp), i_g + t_rng - 1] =  1.0
        ineq_rows.append(A_gramp_dn)
        ineq_lo.append(np.full(n_ramp, -np.inf))
        ineq_hi.append(np.full(n_ramp, R_GRID))

    # Stack all equality and inequality into one LinearConstraint
    A_ineq = sp_vstack(ineq_rows, format="csc")
    lo_ineq = np.concatenate(ineq_lo)
    hi_ineq = np.concatenate(ineq_hi)

    A_all = sp_vstack([A_eq.tocsc(), A_ineq], format="csc")
    lo_all = np.concatenate([b_eq_lo, lo_ineq])
    hi_all = np.concatenate([b_eq_hi, hi_ineq])

    constraint = LinearConstraint(A_all, lo_all, hi_all)

    return obj, constraint, bounds, integrality, {
        "T": T,
        "i_g": i_g, "i_s": i_s, "i_pc": i_pc, "i_pd": i_pd,
        "i_E": i_E, "i_pkoff": i_pkoff, "i_pkon": i_pkon,
        "i_uc": i_uc, "i_ud": i_ud,
    }


def solve(df: pd.DataFrame | None = None,
          mip_gap: float = 0.005,
          time_limit_s: float = 600.0) -> HardResult:
    """
    Build and solve the hard MILP. If df is None, calls build_dataset() automatically.

    mip_gap      relative optimality gap at which the solver stops (default 0.5%)
    time_limit_s wall-clock cutoff in seconds (default 600 s)

    Returns a HardResult with all decision variable arrays and cost components.
    """
    if df is None:
        df = build_dataset()

    obj, constraint, bounds, integrality, idx = _build_milp(df)

    options = {
        "mip_rel_gap": mip_gap,
        "time_limit":  time_limit_s,
        "disp":        False,
    }

    result = milp(
        c           = obj,
        constraints = constraint,
        integrality = integrality,
        bounds      = bounds,
        options     = options,
    )

    if result.status not in (0, 3):
        # 0 = optimal, 3 = time limit hit but a feasible solution exists
        raise RuntimeError(f"MILP did not find a feasible solution — status {result.status}: {result.message}")

    T  = idx["T"]
    x  = result.x

    g      = x[idx["i_g"]    : idx["i_g"]    + T]
    s      = x[idx["i_s"]    : idx["i_s"]    + T]
    pc     = x[idx["i_pc"]   : idx["i_pc"]   + T]
    pd_    = x[idx["i_pd"]   : idx["i_pd"]   + T]
    E      = x[idx["i_E"]    : idx["i_E"]    + T + 1]
    pk_off = x[idx["i_pkoff"]: idx["i_pkoff"] + 12]
    pk_on  = x[idx["i_pkon"] : idx["i_pkon"]  + 12]
    uc     = x[idx["i_uc"]   : idx["i_uc"]   + T]
    ud     = x[idx["i_ud"]   : idx["i_ud"]   + T]

    energy_bill = float(df["c"].values @ g * DT)
    demand_bill = float(D_OFF * pk_off.sum() + D_ON * pk_on.sum())
    wear_bill   = float(C_DEG * (pc + pd_).sum() * DT)

    # HiGHS reports the MIP gap in the fun field's companion; fall back to nan if absent
    gap = getattr(result, "mip_gap", float("nan"))

    return HardResult(
        status      = result.message,
        mip_gap     = gap,
        cost        = float(result.fun),
        energy_bill = energy_bill,
        demand_bill = demand_bill,
        wear_bill   = wear_bill,
        g=g, s=s, pc=pc, pd=pd_, E=E,
        uc=uc, ud=ud,
        pk_off=pk_off, pk_on=pk_on,
        df=df,
    )


def save(res: HardResult, path: Path | None = None) -> Path:
    """Pickle the HardResult to disk. Returns the path written."""
    out = Path(path) if path else RESULTS_DIR / "hard_result.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"Hard result saved → {out}")
    return out


def load(path: Path | None = None) -> HardResult:
    """Load a previously saved HardResult."""
    src = Path(path) if path else RESULTS_DIR / "hard_result.pkl"

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # pickle saved under __main__ when run as a script; remap to this module
            if module == "__main__":
                module = "hard_model"
            return super().find_class(module, name)

    with open(src, "rb") as f:
        return _Unpickler(f).load()


if __name__ == "__main__":
    from config import MIP_GAP, TIME_LIMIT_S
    print("Building dataset...")
    df = build_dataset()
    print(f"Solving MILP over {len(df)} hours  (gap={MIP_GAP}, limit={TIME_LIMIT_S}s)...")
    res = solve(df, mip_gap=MIP_GAP, time_limit_s=TIME_LIMIT_S)
    save(res)
