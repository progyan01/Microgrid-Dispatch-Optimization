"""
Builds the master 8,760-row hourly table used by both the easy and hard models.

Columns produced
    L       building load (kW)
    S       available PV output (kW)
    c       grid energy price ($/kWh)
    period  tariff period (1=off-peak, 2=on-peak)
    month   calendar month (1-12)

All numeric parameters are imported from config.py.
"""

import numpy as np
import pandas as pd

from config import (
    LOAD_FILE, SOLAR_FILE,
    P_PV, GAMMA, U0, U1, PV_LOSS,
    PRICE_OFF, PRICE_ON, SUMMER_MONTHS,
)


# FPL GSDT-1 tariff — on-peak hour logic
def _is_on_peak(dt: pd.Timestamp) -> bool:
    """Return True if dt (in America/New_York local time) falls in an on-peak window."""
    if dt.weekday() >= 5:           # Saturday=5, Sunday=6 → always off-peak
        return False
    h = dt.hour
    if dt.month in SUMMER_MONTHS:   # Apr–Oct: 12:00–21:00
        return 12 <= h < 21
    else:                           # Nov–Mar: 06:00–10:00 and 18:00–22:00
        return (6 <= h < 10) or (18 <= h < 22)


def build_price(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Map a UTC/EST hourly index to FPL GSDT-1 prices.

    The tariff is defined in America/New_York local clock time (with DST),
    so we convert each timestamp before checking the on-peak windows.
    The load and PV series are NOT shifted — only the price lookup uses local time.

    Returns a DataFrame with columns: period (1/2), c ($/kWh), month.
    """
    # Localize as EST (fixed offset, matching the load/weather data), then
    # convert to America/New_York so DST is applied for the tariff lookup.
    est   = index.tz_localize("Etc/GMT+5")
    local = est.tz_convert("America/New_York")

    period = np.array([2 if _is_on_peak(t) else 1 for t in local])
    price  = np.where(period == 2, PRICE_ON, PRICE_OFF)
    month  = index.month

    return pd.DataFrame({"period": period, "c": price, "month": month}, index=index)


# Step 1: building load
def load_building() -> pd.Series:
    """
    Read the ComStock 15-minute load, convert to hourly kWh/h = kW.

    The parquet has period-ending timestamps in fixed EST.
    Shifting back 15 min gives period-start labels; resampling to 1 h sums
    four intervals into one hourly kWh value (which equals average kW since dt=1h).
    """
    df = pd.read_parquet(LOAD_FILE, columns=["timestamp", "out.electricity.total.energy_consumption"])
    s  = df.set_index("timestamp")["out.electricity.total.energy_consumption"].astype(float)

    # Period-ending → period-starting timestamps
    s.index = s.index - pd.Timedelta(minutes=15)

    # Sum four 15-min intervals → hourly kWh (= kW for dt=1h)
    load = s.resample("1h").sum()

    assert len(load) == 8760,       f"Expected 8760 load hours, got {len(load)}"
    assert load.isna().sum() == 0,  "NaNs found in load series"
    assert (load >= 0).all(),       "Negative load values found"

    return load


# Step 2: solar PV
def load_solar(load_index: pd.DatetimeIndex) -> pd.Series:
    """
    Read NSRDB weather and compute hourly PV output via the Faiman cell-temperature model.

    The NSRDB CSV has two header rows; data starts on row 3 (0-indexed row 2).
    Timestamps are period-ending at minute 30
    We reconstruct period-start timestamps so they align with the load index.
    """
    weather = pd.read_csv(
        SOLAR_FILE,
        skiprows=2,
        usecols=["Year", "Month", "Day", "Hour", "Minute", "Temperature", "GHI", "Wind Speed"],
    )

    # Build period-start timestamps: subtract 30 min from the period-end label
    dt_index = pd.to_datetime(
        weather[["Year", "Month", "Day", "Hour", "Minute"]]
        .rename(columns={"Year": "year", "Month": "month", "Day": "day",
                         "Hour": "hour", "Minute": "minute"})
    ) - pd.Timedelta(minutes=30)
    weather.index = dt_index

    ghi   = weather["GHI"].astype(float)          # W/m²
    t_air = weather["Temperature"].astype(float)  # °C
    wind  = weather["Wind Speed"].astype(float)   # m/s

    # Faiman cell temperature model (pvlib default coefficients)
    t_cell = t_air + ghi / (U0 + U1 * wind)

    # PV output (kW); clip negatives caused by tiny float issues at GHI=0
    pv = P_PV * (ghi / 1000.0) * (1.0 - GAMMA * (t_cell - 25.0)) * (1.0 - PV_LOSS)
    pv = pv.clip(lower=0.0)
    pv.name = "S"

    # Align to load index (both are period-start EST hourly)
    pv = pv.reindex(load_index)
    assert pv.isna().sum() == 0, "PV series has gaps after alignment with load index"

    return pv


# Main pipeline
def build_dataset() -> pd.DataFrame:
    """
    Assemble the 8,760-row master table with columns: L, S, c, period, month.
    Runs assertions to catch data issues early.
    """
    load = load_building()          # kW, hourly, EST period-start
    pv   = load_solar(load.index)   # kW, hourly, aligned to load
    tar  = build_price(load.index)  # price, period, month

    df = pd.concat([load.rename("L"), pv, tar], axis=1)

    # Sanity checks
    assert len(df) == 8760,                     "Dataset must have exactly 8,760 rows"
    assert not df.isna().any().any(),            "Dataset contains NaN values"
    assert (df["L"] >= 0).all(),                "Negative load values"
    assert (df["S"] >= 0).all(),                "Negative PV output"
    assert df["c"].nunique() == 2,              "Expected exactly 2 price levels"
    assert (df["period"].isin([1, 2])).all(),   "Period must be 1 or 2"
    assert (df["period"] == 2).sum() > 2000,    "Too few on-peak hours"

    annual_load = df["L"].sum()
    assert 690_000 < annual_load < 715_000, (
        f"Annual load {annual_load:.0f} kWh is outside expected range ~701,806 kWh"
    )

    return df


# Diagnostics when run directly
if __name__ == "__main__":
    df = build_dataset()

    print("Dataset shape:", df.shape)
    print("\nColumn summary:")
    print(df.describe().round(3))

    print(f"\nAnnual load:        {df['L'].sum():,.0f} kWh")
    print(f"Annual PV output:   {df['S'].sum():,.0f} kWh")
    print(f"On-peak hours:      {(df['period'] == 2).sum()}")
    print(f"Price levels:       {sorted(df['c'].unique())}")
    print(f"Peak load:          {df['L'].max():.1f} kW")
    print(f"Peak PV:            {df['S'].max():.1f} kW")
    print("\nFirst 3 rows:")
    print(df.head(3))
