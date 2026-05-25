"""
Analytics Engine — SMA, Gamma Walls, OI Outliers, Earnings
All pure-Python / pandas — no external data dependencies.
Input is structured data already fetched by SchwabClient.
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# ── Simple Moving Averages ──────────────────────────────────────────────────

def compute_smas(candles: list[dict], periods: list[int] = [8, 20, 50]) -> pd.DataFrame:
    """
    Convert Schwab pricehistory candles → DataFrame with SMA columns.

    candles: list of {datetime, open, high, low, close, volume}
    Returns DataFrame indexed by datetime with columns:
        open, high, low, close, volume, sma_8, sma_20, sma_50
    """
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.sort_values("datetime").set_index("datetime")

    for p in periods:
        df[f"sma_{p}"] = df["close"].rolling(window=p, min_periods=p).mean()

    return df


def latest_smas(candles: list[dict], periods: list[int] = [8, 20, 50]) -> dict:
    """Return the most recent SMA values for each period."""
    df = compute_smas(candles, periods)
    if df.empty:
        return {f"sma_{p}": None for p in periods}
    last = df.iloc[-1]
    return {f"sma_{p}": round(float(last[f"sma_{p}"]), 2) if not pd.isna(last[f"sma_{p}"]) else None
            for p in periods}


def sma_signal(price: float, smas: dict) -> str:
    """
    Quick directional signal based on price vs SMAs.
    Returns: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    """
    vals = [v for v in [smas.get("sma_8"), smas.get("sma_20"), smas.get("sma_50")] if v]
    if not vals:
        return "NEUTRAL"
    above = sum(1 for v in vals if price > v)
    if above == len(vals):
        return "BULLISH"
    elif above == 0:
        return "BEARISH"
    return "NEUTRAL"


# ── Options Chain Parsing ───────────────────────────────────────────────────

def parse_option_chain(chain_data: dict) -> pd.DataFrame:
    """
    Flatten a Schwab /chains response into a tidy DataFrame.

    Columns: expiry, strike, put_call, bid, ask, last, volume,
             open_interest, delta, gamma, theta, vega, iv,
             in_the_money, days_to_expiry
    """
    rows = []
    underlying_price = chain_data.get("underlyingPrice", 0)
    today = datetime.today()

    for side in ("callExpDateMap", "putExpDateMap"):
        put_call = "CALL" if side == "callExpDateMap" else "PUT"
        for exp_key, strikes in chain_data.get(side, {}).items():
            # exp_key format: "2024-05-17:5"  (date:daysToExp)
            try:
                exp_date_str = exp_key.split(":")[0]
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
                dte = (exp_date - today).days
            except Exception:
                dte = 0
                exp_date_str = exp_key

            for strike_str, contracts in strikes.items():
                for c in contracts:
                    rows.append({
                        "expiry":        exp_date_str,
                        "dte":           dte,
                        "strike":        float(strike_str),
                        "put_call":      put_call,
                        "bid":           c.get("bid", 0),
                        "ask":           c.get("ask", 0),
                        "last":          c.get("last", 0),
                        "mark":          c.get("mark", 0),
                        "volume":        c.get("totalVolume", 0),
                        "open_interest": c.get("openInterest", 0),
                        "delta":         c.get("delta", 0),
                        "gamma":         c.get("gamma", 0),
                        "theta":         c.get("theta", 0),
                        "vega":          c.get("vega", 0),
                        "iv":            c.get("volatility", 0),
                        "in_the_money":  c.get("inTheMoney", False),
                        "underlying_price": underlying_price,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ── Gamma Wall Detection ────────────────────────────────────────────────────

def detect_gamma_walls(
    chain_df: pd.DataFrame,
    expiry: Optional[str] = None,
    z_threshold: float = 1.5,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify strikes where gamma exposure (GEX = gamma × OI × 100) is
    an outlier vs all other strikes — likely dealer hedging pressure levels.

    Parameters
    ----------
    chain_df      : Output of parse_option_chain()
    expiry        : Filter to a single expiry (None = nearest non-zero)
    z_threshold   : Z-score cutoff to flag as outlier
    top_n         : Max rows returned

    Returns DataFrame with columns:
        strike, put_call, gamma, open_interest, gex, gex_net, z_score, label
    """
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()

    # Default to nearest expiry
    if expiry is None:
        near_exps = df[df["dte"] >= 0]["dte"].sort_values().unique()
        expiry = df[df["dte"] == near_exps[0]]["expiry"].iloc[0] if len(near_exps) > 0 else None

    if expiry:
        df = df[df["expiry"] == expiry].copy()

    if df.empty:
        return pd.DataFrame()

    # Gamma Exposure: calls are positive (dealers long gamma), puts negative
    df["gex"] = df.apply(
        lambda r: r["gamma"] * r["open_interest"] * 100 * (1 if r["put_call"] == "CALL" else -1),
        axis=1,
    )

    # Aggregate net GEX by strike
    gex_by_strike = (
        df.groupby("strike")["gex"]
        .sum()
        .reset_index()
        .rename(columns={"gex": "gex_net"})
    )

    # Z-score of absolute net GEX
    gex_by_strike["abs_gex"] = gex_by_strike["gex_net"].abs()
    mu   = gex_by_strike["abs_gex"].mean()
    std  = gex_by_strike["abs_gex"].std()
    if std == 0:
        return pd.DataFrame()

    gex_by_strike["z_score"] = (gex_by_strike["abs_gex"] - mu) / std
    outliers = gex_by_strike[gex_by_strike["z_score"] >= z_threshold].copy()
    outliers["label"] = outliers["gex_net"].apply(
        lambda x: "🟢 CALL WALL" if x > 0 else "🔴 PUT WALL"
    )
    outliers["expiry"] = expiry

    # Attach original greeks
    side_gamma = df.groupby(["strike", "put_call"]).agg(
        gamma=("gamma", "mean"),
        open_interest=("open_interest", "sum"),
        volume=("volume", "sum"),
    ).reset_index()
    result = outliers.merge(
        side_gamma.groupby("strike").apply(
            lambda g: g.set_index("put_call")["gamma"].to_dict()
        ).reset_index(name="gamma_by_side"),
        on="strike", how="left"
    )

    result = result.sort_values("abs_gex", ascending=False).head(top_n)
    return result[["strike", "expiry", "gex_net", "z_score", "label"]].reset_index(drop=True)


# ── Open Interest Outlier Detection ────────────────────────────────────────

def detect_oi_outliers(
    chain_df: pd.DataFrame,
    expiry: Optional[str] = None,
    z_threshold: float = 1.5,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Flag strikes where total open interest (calls + puts combined) is
    statistically anomalous — key support / resistance / pinning levels.

    Returns DataFrame sorted by total_oi descending with z_score.
    """
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()
    if expiry:
        df = df[df["expiry"] == expiry]

    if df.empty:
        return pd.DataFrame()

    oi_by_strike = (
        df.groupby(["strike", "put_call"])["open_interest"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["CALL", "PUT"]:
        if col not in oi_by_strike.columns:
            oi_by_strike[col] = 0

    oi_by_strike["total_oi"] = oi_by_strike["CALL"] + oi_by_strike["PUT"]
    oi_by_strike["call_oi"]  = oi_by_strike["CALL"]
    oi_by_strike["put_oi"]   = oi_by_strike["PUT"]
    oi_by_strike["pcr"]      = oi_by_strike.apply(
        lambda r: round(r["put_oi"] / r["call_oi"], 2) if r["call_oi"] > 0 else 999,
        axis=1,
    )

    mu  = oi_by_strike["total_oi"].mean()
    std = oi_by_strike["total_oi"].std()
    if std == 0:
        return pd.DataFrame()

    oi_by_strike["z_score"] = (oi_by_strike["total_oi"] - mu) / std
    outliers = oi_by_strike[oi_by_strike["z_score"] >= z_threshold].copy()
    outliers["label"] = outliers["pcr"].apply(
        lambda x: "🔴 PUT HEAVY" if x > 1.5 else ("🟢 CALL HEAVY" if x < 0.67 else "⚪ BALANCED")
    )

    return (
        outliers[["strike", "call_oi", "put_oi", "total_oi", "pcr", "z_score", "label"]]
        .sort_values("total_oi", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Combined Levels Summary ─────────────────────────────────────────────────

def get_key_levels(
    chain_df: pd.DataFrame,
    underlying_price: float,
    expiry: Optional[str] = None,
) -> dict:
    """
    One-stop summary: gamma walls + OI concentrations + nearest levels.

    Returns:
    {
      "gamma_walls": DataFrame,
      "oi_outliers": DataFrame,
      "max_pain": float,
      "nearest_resistance": float,
      "nearest_support": float,
    }
    """
    gamma_walls = detect_gamma_walls(chain_df, expiry)
    oi_outliers = detect_oi_outliers(chain_df, expiry)

    # Max pain = strike where total option premium decay is greatest
    max_pain = _compute_max_pain(chain_df, expiry)

    # Nearest resistance (call wall above price)
    resistance = None
    support    = None
    if not gamma_walls.empty:
        calls_above = gamma_walls[
            (gamma_walls["gex_net"] > 0) &
            (gamma_walls["strike"] > underlying_price)
        ]
        puts_below = gamma_walls[
            (gamma_walls["gex_net"] < 0) &
            (gamma_walls["strike"] < underlying_price)
        ]
        if not calls_above.empty:
            resistance = float(calls_above.sort_values("strike")["strike"].iloc[0])
        if not puts_below.empty:
            support = float(puts_below.sort_values("strike", ascending=False)["strike"].iloc[0])

    return {
        "gamma_walls":        gamma_walls,
        "oi_outliers":        oi_outliers,
        "max_pain":           max_pain,
        "nearest_resistance": resistance,
        "nearest_support":    support,
    }


def _compute_max_pain(chain_df: pd.DataFrame, expiry: Optional[str] = None) -> Optional[float]:
    """
    Max pain = strike where total OTM option value is minimized.
    Fully vectorized via numpy broadcasting — O(strikes) not O(strikes × rows).
    """
    if chain_df.empty:
        return None

    df = chain_df.copy()
    if expiry:
        df = df[df["expiry"] == expiry]

    calls = df[df["put_call"] == "CALL"].groupby("strike")["open_interest"].sum()
    puts  = df[df["put_call"] == "PUT"].groupby("strike")["open_interest"].sum()

    strikes = np.sort(np.union1d(
        calls.index.values if not calls.empty else np.array([]),
        puts.index.values  if not puts.empty  else np.array([]),
    ))
    if len(strikes) == 0:
        return None

    # Broadcasting: test_strikes[:, None] vs contract_strikes[None, :]
    if not calls.empty:
        call_pain = np.maximum(0, strikes[:, None] - calls.index.values[None, :]) \
                    * calls.values[None, :] * 100
    else:
        call_pain = np.zeros((len(strikes), 1))

    if not puts.empty:
        put_pain = np.maximum(0, puts.index.values[None, :] - strikes[:, None]) \
                   * puts.values[None, :] * 100
    else:
        put_pain = np.zeros((len(strikes), 1))

    return float(strikes[(call_pain.sum(axis=1) + put_pain.sum(axis=1)).argmin()])


# ── Earnings Dates ──────────────────────────────────────────────────────────

def get_earnings_dates(symbols: list[str]) -> dict:
    """
    Fetch next earnings dates via yfinance (free, no auth required).
    Returns: {symbol: {"date": "YYYY-MM-DD", "estimate_eps": float, "days_away": int}}
    """
    try:
        import yfinance as yf
    except ImportError:
        return {s: {"date": "yfinance not installed", "days_away": None} for s in symbols}

    result = {}
    today = datetime.today().date()

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and "Earnings Date" in cal:
                raw = cal["Earnings Date"]
                # yfinance returns list or single date
                if isinstance(raw, list) and raw:
                    ed = raw[0]
                elif hasattr(raw, "date"):
                    ed = raw
                else:
                    ed = None

                if ed:
                    ed_date = ed.date() if hasattr(ed, "date") else ed
                    days_away = (ed_date - today).days
                    result[symbol] = {
                        "date":      str(ed_date),
                        "days_away": days_away,
                        "flag":      "🔥 <7 days" if days_away < 7 else (
                                     "⚠️ <14 days" if days_away < 14 else "📅 Upcoming"
                                     ) if days_away >= 0 else "✅ Past",
                    }
                else:
                    result[symbol] = {"date": "N/A", "days_away": None, "flag": ""}
            else:
                result[symbol] = {"date": "N/A", "days_away": None, "flag": ""}
        except Exception as e:
            result[symbol] = {"date": f"Error: {e}", "days_away": None, "flag": ""}

    return result


# ── Greeks Summary for a Position ──────────────────────────────────────────

def position_greeks(positions: list[dict], chain_df: pd.DataFrame) -> list[dict]:
    """
    Enrich position records with live greeks from the options chain.
    Matches on strike, expiry, and put/call type.
    """
    if chain_df.empty:
        return positions

    enriched = []
    for pos in positions:
        if pos.get("asset_type") != "OPTION":
            enriched.append(pos)
            continue

        match = chain_df[
            (chain_df["strike"] == pos.get("strike")) &
            (chain_df["expiry"].str.startswith(str(pos.get("expiry", ""))[:10])) &
            (chain_df["put_call"] == pos.get("put_call", ""))
        ]

        if not match.empty:
            row = match.iloc[0]
            qty = pos.get("quantity", 0)
            mult = pos.get("multiplier", 100)
            pos = {
                **pos,
                "delta":        round(row["delta"] * qty * mult, 2),
                "gamma":        round(row["gamma"] * qty * mult, 4),
                "theta":        round(row["theta"] * qty * mult, 2),
                "vega":         round(row["vega"]  * qty * mult, 2),
                "iv":           round(row["iv"], 1),
                "mark":         round(row["mark"], 2),
                "live_value":   round(row["mark"] * abs(qty) * mult, 2),
            }
        enriched.append(pos)
    return enriched
