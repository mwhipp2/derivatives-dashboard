"""
yfinance Drop-in Client
=======================
Implements the same interface as SchwabClient using Yahoo Finance data.
No API keys or authentication required — works immediately.

Key design decisions:
  - Quotes use .history(period="1d") — reliable, works for any symbol
  - Options chain loads ONE expiry at a time for speed
  - Greeks computed via Black-Scholes where yfinance doesn't supply them
  - get_expirations() returns the full list fast (single yfinance call)
"""

import math
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
from scipy.stats import norm

# ── TTL cache ─────────────────────────────────────────────────────────────────
# Keyed by arbitrary string; value is (result, monotonic_timestamp).
# Avoids hitting yfinance on every 30-second Dash refresh tick.

_CACHE: dict = {}

def _cache_get(key: str, ttl: float):
    entry = _CACHE.get(key)
    if entry and (_time.monotonic() - entry[1] < ttl):
        return entry[0], True
    return None, False

def _cache_set(key: str, val):
    _CACHE[key] = (val, _time.monotonic())


# ── NaN-safe converters ───────────────────────────────────────────────────────
# Python's `or` does NOT handle float NaN — `float('nan') or 0` returns nan
# because NaN is truthy. Use these everywhere yfinance data is coerced.

def _nf(val, default: float = 0.0) -> float:
    """float-safe: returns default on None / NaN / non-numeric."""
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default

def _ni(val) -> int:
    """int-safe: returns 0 on None / NaN / non-numeric."""
    try:
        f = float(val)
        return 0 if math.isnan(f) or math.isinf(f) else int(f)
    except (TypeError, ValueError):
        return 0


# ── Black-Scholes greeks ─────────────────────────────────────────────────────

def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> dict:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if flag == "c":
            delta = norm.cdf(d1)
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega  = S * norm.pdf(d1) * math.sqrt(T) / 100
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega":  round(vega,  4),
        }
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def _spot_price(symbol: str, ticker=None) -> float:
    """
    Spot price with three fallback tiers:
      1. fast_info.last_price  — single lightweight metadata call, no OHLCV
      2. 1-day 1-minute history — works during market hours
      3. 5-day daily history   — always works, uses prior close after hours
    Accepts an existing Ticker to avoid creating a duplicate object.
    """
    t = ticker or yf.Ticker(symbol)

    # Tier 1: fast_info (yfinance ≥ 0.2)
    try:
        fi = t.fast_info
        price = _nf(getattr(fi, "last_price", None))
        if price <= 0:
            price = _nf(getattr(fi, "previous_close", None))
        if price > 0:
            return round(price, 2)
    except Exception:
        pass

    # Tier 2: intraday history
    try:
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass

    # Tier 3: daily history
    try:
        hist = t.history(period="5d", interval="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass

    return 0.0


# ── Client ────────────────────────────────────────────────────────────────────

class YFinanceClient:

    def is_authenticated(self) -> bool:
        return True

    def authenticate(self):
        pass

    # ── Quotes ───────────────────────────────────────────────────────────
    def get_quotes(self, symbols: list[str]) -> dict:
        """
        Fetch last close + day change for a list of symbols.
        Fetches all symbols concurrently (one thread per symbol) with a 12s
        wall-clock timeout; caches results for 60 s to reduce yfinance calls.
        """
        result = {}

        def _fetch_one(sym: str) -> tuple[str, dict]:
            cache_key = f"quote:{sym}"
            cached, hit = _cache_get(cache_key, ttl=60)
            if hit:
                return sym, cached
            try:
                hist = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=True)
                if hist.empty:
                    raise ValueError("no history")
                last = round(float(hist["Close"].iloc[-1]), 2)
                prev = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else last
                chg  = round(last - prev, 4)
                pct  = round((chg / prev * 100) if prev else 0, 4)
                data = {"quote": {
                    "lastPrice":                last,
                    "bidPrice":                 round(last * 0.999, 2),
                    "askPrice":                 round(last * 1.001, 2),
                    "netChange":                chg,
                    "netPercentChangeInDouble": pct,
                    "mark":                     last,
                }}
                _cache_set(cache_key, data)
                return sym, data
            except Exception as e:
                return sym, {"quote": {
                    "lastPrice": 0, "netChange": 0,
                    "netPercentChangeInDouble": 0, "_error": str(e),
                }}

        n = max(1, min(len(symbols), 10))
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_fetch_one, s): s for s in symbols}
            try:
                for fut in as_completed(futures, timeout=12):
                    sym, data = fut.result()
                    result[sym] = data
            except Exception:
                pass  # timeout or unexpected error — fall through to blank-entry fill

        # Fill any symbols that timed out or errored before completing
        for sym in symbols:
            if sym not in result:
                result[sym] = {"quote": {"lastPrice": 0, "netChange": 0,
                                         "netPercentChangeInDouble": 0,
                                         "_error": "timeout"}}
        return result

    # ── Price history ─────────────────────────────────────────────────────
    def get_price_history(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 3,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> dict:
        cache_key = f"hist:{symbol}:{period_type}:{period}:{frequency_type}"
        cached, hit = _cache_get(cache_key, ttl=300)  # 5-minute cache
        if hit:
            return cached

        period_map = {
            ("day",   1): "5d",  ("day",   5): "5d",
            ("month", 1): "1mo", ("month", 2): "2mo", ("month", 3): "3mo",
            ("month", 6): "6mo",
            ("year",  1): "1y",  ("year",  2): "2y",  ("year",  5): "5y",
            ("ytd",   1): "ytd",
        }
        yf_period   = period_map.get((period_type, period), "3mo")
        freq_map    = {"minute": "1m", "daily": "1d", "weekly": "1wk", "monthly": "1mo"}
        yf_interval = freq_map.get(frequency_type, "1d")

        try:
            df = yf.Ticker(symbol).history(period=yf_period, interval=yf_interval, auto_adjust=True)
            candles = []
            for ts, row in df.iterrows():
                candles.append({
                    "datetime": int(ts.timestamp() * 1000),
                    "open":     round(float(row["Open"]),  4),
                    "high":     round(float(row["High"]),  4),
                    "low":      round(float(row["Low"]),   4),
                    "close":    round(float(row["Close"]), 4),
                    "volume":   int(row["Volume"]),
                })
            result = {"candles": candles, "symbol": symbol}
            _cache_set(cache_key, result)
            return result
        except Exception as e:
            return {"candles": [], "symbol": symbol, "_error": str(e)}

    # ── Expiration dates (fast, single call) ──────────────────────────────
    def get_expirations(self, symbol: str) -> list[str]:
        """
        Return sorted list of available expiry date strings "YYYY-MM-DD".
        Very fast — single yfinance metadata call, no chain data fetched.
        """
        cache_key = f"exps:{symbol}"
        cached, hit = _cache_get(cache_key, ttl=600)  # 10-minute cache
        if hit:
            return cached
        try:
            exps = yf.Ticker(symbol).options   # tuple of date strings
            today = datetime.today().date()
            result = [e for e in exps
                      if datetime.strptime(e, "%Y-%m-%d").date() >= today]
            _cache_set(cache_key, result)
            return result
        except Exception:
            return []

    # ── Single-expiry option chain ────────────────────────────────────────
    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "ALL",
        strike_count: int = 40,
        include_underlying: bool = True,
        strategy: str = "SINGLE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        expiry: Optional[str] = None,     # ← preferred: fetch just this date
    ) -> dict:
        """
        Fetch option chain for a single expiry (preferred) or the nearest
        available expiry. Returns a Schwab-shaped dict.
        """
        cache_key = f"chain:{symbol}:{expiry or 'nearest'}"
        cached, hit = _cache_get(cache_key, ttl=120)  # 2-minute cache
        if hit:
            return cached

        last_error = ""
        for attempt in range(2):   # retry once on transient failure
            try:
                ticker = yf.Ticker(symbol)
                spot   = _spot_price(symbol, ticker=ticker)
                today  = datetime.today()
                r      = 0.053

                all_expiries = list(ticker.options or [])
                if not all_expiries:
                    return self._empty_chain(symbol, spot, error="no expiries available")

                # Pick expiry: exact match → fuzzy ±1 day match → nearest available
                target_exp = None
                if expiry:
                    if expiry in all_expiries:
                        target_exp = expiry
                    else:
                        # Try ±1 day (handles Saturday settlement listings, etc.)
                        from datetime import date as _date, timedelta as _td
                        try:
                            req = _date.fromisoformat(expiry)
                            for delta in (1, -1, 2, -2):
                                alt = (req + _td(days=delta)).isoformat()
                                if alt in all_expiries:
                                    target_exp = alt
                                    break
                        except ValueError:
                            pass
                if target_exp is None:
                    target_exp = all_expiries[0]

                exp_dt  = datetime.strptime(target_exp, "%Y-%m-%d")
                dte     = max((exp_dt - today).days, 0)
                T       = max(dte / 365, 1 / 365)
                exp_key = f"{target_exp}:{dte}"

                chain = ticker.option_chain(target_exp)

                call_map: dict = {}
                put_map:  dict = {}

                for side, df_raw, target_map in [
                    ("CALL", chain.calls, call_map),
                    ("PUT",  chain.puts,  put_map),
                ]:
                    if df_raw is None or df_raw.empty:
                        continue

                    # Drop rows with NaN strikes before any processing
                    df_raw = df_raw.dropna(subset=["strike"])

                    # Limit to strike_count strikes nearest ATM
                    if spot > 0:
                        strikes_near = sorted(
                            df_raw["strike"].tolist(),
                            key=lambda k: abs(k - spot)
                        )[:strike_count]
                        df_f = df_raw[df_raw["strike"].isin(strikes_near)].copy()
                    else:
                        df_f = df_raw.copy()

                    strikes_dict: dict = {}
                    for _, row in df_f.iterrows():
                        K   = _nf(row["strike"])
                        if K <= 0:
                            continue
                        iv  = _nf(row.get("impliedVolatility"))
                        bid  = _nf(row.get("bid"))
                        ask  = _nf(row.get("ask"))
                        last = _nf(row.get("lastPrice"))
                        mark = round((bid + ask) / 2, 2) if (bid > 0 and ask > 0) else last

                        greeks = _bs_greeks(spot, K, T, r, iv,
                                            "c" if side == "CALL" else "p")

                        strikes_dict[str(K)] = [{
                            "bid":            bid,
                            "ask":            ask,
                            "last":           last,
                            "mark":           mark,
                            "totalVolume":    _ni(row.get("volume")),
                            "openInterest":   _ni(row.get("openInterest")),
                            "volatility":     round(iv * 100, 2),
                            "delta":          greeks["delta"],
                            "gamma":          greeks["gamma"],
                            "theta":          greeks["theta"],
                            "vega":           greeks["vega"],
                            "inTheMoney":     bool(row.get("inTheMoney") or False),
                            "putCall":        side,
                            "strikePrice":    K,
                            "expirationDate": target_exp,
                        }]

                    if strikes_dict:
                        target_map[exp_key] = strikes_dict

                result = {
                    "symbol":          symbol.upper(),
                    "underlyingPrice": round(spot, 2),
                    "callExpDateMap":  call_map,
                    "putExpDateMap":   put_map,
                    "status":          "SUCCESS",
                    "_source":         "yfinance",
                    "_expiry":         target_exp,
                }
                _cache_set(cache_key, result)
                return result

            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    _time.sleep(0.6)   # brief backoff before retry

        return self._empty_chain(symbol, 0, error=last_error)

    def _empty_chain(self, symbol: str, spot: float, error: str = "") -> dict:
        return {
            "symbol":          symbol,
            "underlyingPrice": spot,
            "callExpDateMap":  {},
            "putExpDateMap":   {},
            "status":          "ERROR",
            "_error":          error,
        }

    # ── Account stubs ─────────────────────────────────────────────────────
    def get_accounts(self) -> list[dict]:
        return []

    def get_positions(self) -> list[dict]:
        return []

    def get_unique_underlyings(self) -> list[str]:
        return []
