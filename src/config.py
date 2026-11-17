"""
Single source of truth for every numeric parameter and file path in the project.
Both easy_model.py and hard_model.py import from here; data_prep.py does too.

Sources
  PV model      : pvlib defaults (Faiman 1998, doi:10.1016/S0038-092X(98)00054-8)
  Battery cost  : NREL ATB 2024, Moderate scenario, utility-scale Li-ion
                  (atb.nrel.gov/electricity/2024/utility-scale_battery_storage)
  Cycle life    : NREL ATB 2024; 3,000–5,000 cycles typical for LFP at 80–90% DoD
  Self-discharge: ~2%/month industry consensus for LFP pack incl. BMS parasitic draw
                  (Linden & Reddy, Handbook of Batteries; manufacturer spec sheets)
  Inverter P_min: 5–10% of rated power, typical efficiency-curve cutoff for commercial
                  3-phase string inverters (SMA, Fronius efficiency curves)
  Tariff        : FPL GSDT-1, OpenEI URDB rate ID 66a911f6ad5e5f4d84062c74
"""

import math
from pathlib import Path


# Paths
ROOT        = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR    = ROOT / "data"                         # raw input files
RESULTS_DIR = ROOT / "results"                      # solver outputs

LOAD_FILE  = DATA_DIR / "26194-0.parquet"   # ComStock 15-min building load
SOLAR_FILE = DATA_DIR / "solardata.csv"     # NSRDB hourly weather for Miami 2018


# Time resolution
DT = 1.0    # time-step duration (hours); kept as a float so formulas stay general
T  = 8760   # total hours in 2018 (non-leap year: 365 × 24)


# PV system
P_PV    = 200.0   # installed PV capacity (kW); tune across 150–250 kW
GAMMA   = 0.004   # temperature derating coefficient for crystalline Si (per °C); pvlib default
U0      = 25.0    # Faiman model still-air heat-loss coefficient (W m⁻² K⁻¹); pvlib default
U1      = 6.84    # Faiman model wind heat-loss coefficient (W s m⁻³ K⁻¹); pvlib default
PV_LOSS = 0.12    # overall system loss fraction covering soiling, wiring, and inverter


# Battery — shared by both models
E_CAP = 400.0   # usable energy capacity (kWh); tune
P_BAR = 100.0   # maximum charge or discharge power (kW)

# Round-trip efficiency 90 %; split symmetrically so ETA_C × ETA_D = 0.90
ETA_C = math.sqrt(0.90)   # one-way charge efficiency; ≈ 0.9487
ETA_D = math.sqrt(0.90)   # one-way discharge efficiency; ≈ 0.9487

SOC_MIN = 0.10              # lower state-of-charge limit (fraction of E_CAP)
SOC_MAX = 0.90              # upper state-of-charge limit (fraction of E_CAP)
E_MIN   = SOC_MIN * E_CAP  # minimum stored energy (kWh); = 40 kWh
E_MAX   = SOC_MAX * E_CAP  # maximum stored energy (kWh); = 360 kWh
E_INIT  = 0.50 * E_CAP     # initial and final stored energy (kWh); enforces cyclic boundary

# ~2 %/month LFP pack self-discharge including BMS parasitic draw (Linden & Reddy, Handbook of Batteries)
LAMBDA = 3e-5   # self-discharge rate (fraction of stored energy lost per hour)


# Battery — hard model only
C_B      = 310.0                 # battery capital cost ($/kWh); NREL ATB 2024 Moderate mid-point
N_CYCLES = 4000                  # cycle life at 80–90 % DoD; NREL ATB 2024 LFP mid-point
C_DEG    = C_B / (2 * N_CYCLES)  # throughput wear cost ($/kWh); = C_B / (2 × N_CYCLES) ≈ 0.03875

# 10 % of P_BAR matches the 5–10 % minimum-load efficiency cutoff on commercial 3-phase inverters
P_MIN = 0.10 * P_BAR   # minimum active charge or discharge power (kW); = 10 kW

R_BAT  = 0.50 * P_BAR  # maximum change in battery net power between consecutive hours (kW/h); = 50 kW

# Largest observed hourly load step is ~63.5 kW; 70 kW keeps the MILP feasible
R_GRID = 70.0   # maximum change in grid import between consecutive hours (kW/h); set None to disable


# FPL GSDT-1 tariff — energy prices = rate + adjustment ($/kWh)
PRICE_OFF = 0.01366 + 0.03032   # off-peak energy price ($/kWh); = 0.04398
PRICE_ON  = 0.05421 + 0.03790   # on-peak energy price ($/kWh);  = 0.09211

# Demand charges = rate + adjustment ($/kW-month)
D_OFF = 0.71  + 1.64   # off-peak monthly peak demand charge ($/kW-month); = 2.35
D_ON  = 10.67 + 1.64   # on-peak monthly peak demand charge ($/kW-month);  = 12.31

# Apr–Oct weekdays are on-peak 12:00–21:00; Nov–Mar weekdays 06:00–10:00 and 18:00–22:00
SUMMER_MONTHS = frozenset({4, 5, 6, 7, 8, 9, 10})   # months using the summer on-peak window


# Solver
SOLVER       = "HIGHS"   # CVXPY solver used for both the LP and MILP
MIP_GAP      = 0.005     # relative optimality gap at which the MILP is accepted (0.5 %)
TIME_LIMIT_S = 600       # wall-clock cutoff for the MILP solver (seconds)
