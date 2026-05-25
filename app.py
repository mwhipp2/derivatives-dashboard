"""
Derivatives Trading Dashboard
==============================
Run:  python app.py
Then: http://127.0.0.1:8050

Data source is selected automatically:
  • Schwab API  — if SCHWAB_APP_KEY / SCHWAB_APP_SECRET are set in .env
                  AND a saved token exists (.schwab_tokens.json)
  • yfinance    — fallback; free, no auth, works immediately.
                  Quotes are 15-min delayed; options chain uses
                  Black-Scholes greeks. Positions panel will be empty.
"""

import json
import os
import threading
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from analytics import (
    compute_smas,
    detect_gamma_walls,
    detect_oi_outliers,
    get_earnings_dates,
    get_key_levels,
    parse_option_chain,
    position_greeks,
    sma_signal,
    latest_smas,
)

load_dotenv()

# ── Company name cache ────────────────────────────────────────────────────────
# yf.Ticker().info is one of yfinance's slowest calls (~1-2 s per symbol).
# Cache names for the session so chart reloads don't re-fetch.
_company_names: dict[str, str] = {}

def _get_company_name(symbol: str) -> str:
    if symbol not in _company_names:
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            _company_names[symbol] = info.get("longName") or info.get("shortName") or symbol
        except Exception:
            _company_names[symbol] = symbol
    return _company_names[symbol]

# ── File-based watchlist persistence ─────────────────────────────────────────
# Three-layer resilience:
#   1. Primary file  (watchlist.json)     — written atomically via temp+rename
#   2. Backup file   (watchlist.bak.json) — written after every successful primary write
#   3. URL-triggered page-load callback   — always reads fresh from disk so that
#      browser refresh AND code-reload both restore the full symbol list
#
# dcc.Store uses storage_type="memory" (not localStorage) so Dash fingerprint
# changes never wipe data. The callback layer owns persistence, not the browser.

WATCHLIST_FILE = Path("watchlist.json")
WATCHLIST_BAK  = Path("watchlist.bak.json")
LAST_SYM_FILE  = Path("last_symbol.json")
LAST_SYM_BAK   = Path("last_symbol.bak.json")


def _load_watchlist() -> list:
    """
    Load order:
      1. Primary file (watchlist.json)
      2. Backup file  (watchlist.bak.json)
      3. WATCHLIST env var — used on cloud hosts where the filesystem is ephemeral
         Set it in Render's environment variables as: ORCL,PLTR,SOFI,BAC,NKE,WMT,NVDA
    """
    for path in (WATCHLIST_FILE, WATCHLIST_BAK):
        try:
            if not path.exists():
                continue
            raw = path.read_text().strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, list):
                symbols = [s.strip().upper() for s in data
                           if isinstance(s, str) and s.strip()]
                if symbols:
                    return symbols
        except Exception:
            continue
    # Cloud fallback: WATCHLIST env var
    env = os.getenv("WATCHLIST", "")
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    return []


def _save_watchlist(lst: list):
    """Atomic write (temp→rename) so a crash mid-write never corrupts the file."""
    payload = json.dumps(lst)
    tmp = WATCHLIST_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(payload)
        tmp.replace(WATCHLIST_FILE)          # atomic on POSIX / near-atomic on Windows
        try:
            WATCHLIST_BAK.write_text(payload)  # best-effort backup
        except Exception:
            pass
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_last_symbol() -> str | None:
    for path in (LAST_SYM_FILE, LAST_SYM_BAK):
        try:
            if not path.exists():
                continue
            raw = path.read_text().strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, str) and data.strip():
                return data.strip().upper()
        except Exception:
            continue
    return None


def _save_last_symbol(sym: str | None):
    payload = json.dumps(sym)
    tmp = LAST_SYM_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(payload)
        tmp.replace(LAST_SYM_FILE)
        try:
            LAST_SYM_BAK.write_text(payload)
        except Exception:
            pass
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


_initial_watchlist   = _load_watchlist()
_initial_last_symbol = _load_last_symbol()

# ── Auto-select data source ──────────────────────────────────────────────────
def _build_client():
    """
    Use SchwabClient if credentials + a saved token are present.
    Otherwise fall back to YFinanceClient (no auth required).
    """
    has_creds  = bool(os.getenv("SCHWAB_APP_KEY")) and bool(os.getenv("SCHWAB_APP_SECRET"))
    has_tokens = Path(".schwab_tokens.json").exists()

    if has_creds:
        from schwab_client import SchwabClient
        c = SchwabClient()
        if has_tokens and c.is_authenticated():
            print("✅  Using Schwab API (live data + account positions)")
            return c, "schwab"
        elif has_creds:
            # Credentials present but no token yet — trigger auth flow
            print("🔐  Schwab credentials found — starting OAuth login…")
            c.authenticate()
            return c, "schwab"

    # Fallback
    from yfinance_client import YFinanceClient
    print("ℹ️   Schwab API not configured — using yfinance (15-min delayed, no positions)")
    return YFinanceClient(), "yfinance"

client, DATA_SOURCE = _build_client()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],   # dark trading theme
    title="Derivatives Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server   # for gunicorn if needed

REFRESH_MS    = 30_000      # 30-second live data refresh
COLORS = {
    "bg":        "#0d1117",
    "card":      "#161b22",
    "border":    "#30363d",
    "green":     "#3fb950",
    "red":       "#f85149",
    "yellow":    "#d29922",
    "blue":      "#58a6ff",
    "muted":     "#8b949e",
    "text":      "#e6edf3",
    "sma8":      "#f7c948",
    "sma20":     "#79c0ff",
    "sma50":     "#ffa657",
    "gamma_pos": "rgba(63,185,80,0.25)",
    "gamma_neg": "rgba(248,81,73,0.25)",
}

# ── Layout ───────────────────────────────────────────────────────────────────
def _card(title: str, content, id_prefix: str = "") -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            title,
            style={"background": COLORS["card"], "color": COLORS["blue"],
                   "fontWeight": "700", "fontSize": "11px", "letterSpacing": "1px",
                   "padding": "5px 10px",
                   "border": f"1px solid {COLORS['border']}"}
        ),
        dbc.CardBody(
            content,
            style={"background": COLORS["bg"], "padding": "8px",
                   "border": f"1px solid {COLORS['border']}"}
        ),
    ], style={"border": f"1px solid {COLORS['border']}", "marginBottom": "8px"})


app.layout = dbc.Container([
    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("⚡ ", style={"color": COLORS["yellow"]}),
                html.Span("DERIVATIVES DASHBOARD", style={
                    "fontSize": "17px", "fontWeight": "900",
                    "letterSpacing": "3px", "color": COLORS["text"]
                }),
                html.Span(
                    " · LIVE  |  " + ("SCHWAB" if DATA_SOURCE == "schwab" else "yfinance (15-min delay)"),
                    style={
                        "fontSize": "11px",
                        "color": COLORS["green"] if DATA_SOURCE == "schwab" else COLORS["yellow"],
                        "marginLeft": "12px", "letterSpacing": "2px",
                    }
                ),
            ], style={"padding": "8px 0 6px 0"}),
        ], width=8),
        dbc.Col([
            html.Div(id="last-update", style={
                "color": COLORS["muted"], "fontSize": "11px",
                "textAlign": "right", "paddingTop": "12px"
            }),
        ], width=4),
    ]),

    # ── Symbol Input Row ─────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(
                    id="symbol-input",
                    placeholder="Add symbol (e.g. NVDA)",
                    debounce=False,
                    style={"background": COLORS["card"], "color": COLORS["text"],
                           "border": f"1px solid {COLORS['border']}", "fontSize": "13px"},
                ),
                dbc.Button("＋ Add", id="add-symbol-btn", color="success", size="sm"),
                dbc.Button("⟳ Refresh Positions", id="refresh-positions-btn",
                           color="secondary", size="sm", style={"marginLeft": "6px"}),
            ]),
        ], width=8),
        dbc.Col([
            dcc.Dropdown(
                id="selected-symbol",
                placeholder="Select symbol to analyze…",
                style={"background": COLORS["card"], "color": COLORS["text"]},
            ),
        ], width=4),
    ], style={"marginBottom": "6px"}),

    # ── Watchlist chips ──────────────────────────────────────────────────────
    html.Div(id="watchlist-chips", style={"marginBottom": "6px"}),

    # ── Quote Bar ────────────────────────────────────────────────────────────
    html.Div(id="quote-bar", style={"marginBottom": "6px"}),

    # ── Row 1: Earnings | Key Levels | Positions ─────────────────────────────
    dbc.Row([
        dbc.Col([
            _card("EARNINGS CALENDAR", html.Div(id="earnings-table")),
        ], width=4),
        dbc.Col([
            _card("KEY LEVELS SUMMARY", html.Div(id="levels-summary")),
        ], width=4),
        dbc.Col([
            _card("OPEN POSITIONS", html.Div(id="positions-table")),
        ], width=4),
    ]),

    # ── Row 2: Price Chart | SMA All-Symbols Table ────────────────────────────
    dbc.Row([
        dbc.Col([
            _card("PRICE CHART · SMA 8 / 20 / 50", [
                html.Div(id="chart-symbol-header", style={"marginBottom": "4px"}),
                dcc.Graph(id="price-chart", config={"displayModeBar": False},
                          style={"height": "300px"}),
                html.Div(id="sma-badges", style={"marginTop": "4px"}),
            ]),
        ], width=7),
        dbc.Col([
            _card("SMA LEVELS · ALL SYMBOLS", html.Div(id="sma-watchlist-table")),
        ], width=5),
    ]),

    # ── Row 3: Gamma Walls | OI Chart ────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            _card("GAMMA WALLS · GEX OUTLIERS", [
                dcc.Dropdown(
                    id="expiry-selector",
                    placeholder="Select expiry…",
                    style={"background": COLORS["card"], "color": "#000",
                           "marginBottom": "6px", "fontSize": "12px"},
                ),
                dcc.Graph(id="gamma-chart", config={"displayModeBar": False},
                          style={"height": "240px"}),
                html.Div(id="gamma-table"),
            ]),
        ], width=6),
        dbc.Col([
            _card("OPEN INTEREST · KEY LEVELS", [
                dcc.Graph(id="oi-chart", config={"displayModeBar": False},
                          style={"height": "240px"}),
                html.Div(id="oi-table"),
            ]),
        ], width=6),
    ]),

    # ── Row 4: Gamma Walls All-Symbols Table ──────────────────────────────────
    dbc.Row([
        dbc.Col([
            _card("GAMMA WALLS · NEXT MONTHLY EXPIRY · ALL SYMBOLS",
                  html.Div(id="gamma-watchlist-table")),
        ], width=12),
    ]),

    # ── State stores ─────────────────────────────────────────────────────────
    # Location fires on every page load — used to restore stores from disk.
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="watchlist-store",    data=_initial_watchlist,   storage_type="memory"),
    dcc.Store(id="last-symbol-store",  data=_initial_last_symbol, storage_type="memory"),
    dcc.Store(id="chain-store",        data={}),
    dcc.Store(id="positions-store",    data=[]),
    dcc.Interval(id="auto-refresh", interval=REFRESH_MS, n_intervals=0),
], fluid=True, style={"background": COLORS["bg"], "minHeight": "100vh", "padding": "0 12px"})


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("watchlist-store", "data"),
    Input("url", "pathname"),                                        # page load / refresh
    Input("add-symbol-btn", "n_clicks"),
    Input("refresh-positions-btn", "n_clicks"),
    Input({"type": "remove-sym-btn", "index": ALL}, "n_clicks"),
    State("symbol-input", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=False,
)
def update_watchlist(_, add_clicks, refresh_clicks, remove_clicks, symbol, current_list):
    prop_id = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    # ── Page load / browser refresh ──────────────────────────────────────────
    # Always read fresh from disk so any code-reload or browser refresh
    # gets the full saved list, not the stale module-level snapshot.
    if "url" in prop_id or not prop_id:
        saved = _load_watchlist()
        return saved if saved else (current_list or [])

    current_list = current_list or []
    new_list = no_update

    if "add-symbol-btn" in prop_id and symbol:
        sym = symbol.upper().strip()
        if sym and sym not in current_list:
            new_list = current_list + [sym]

    elif "refresh-positions-btn" in prop_id:
        try:
            underlyings = client.get_unique_underlyings()
            new_list = list(dict.fromkeys(current_list + underlyings))  # preserves order, dedupes
        except Exception:
            pass

    elif "remove-sym-btn" in prop_id:
        try:
            prop = callback_context.triggered[0]["prop_id"].rsplit(".", 1)[0]
            sym_to_remove = json.loads(prop)["index"]
            new_list = [s for s in current_list if s != sym_to_remove]
        except Exception:
            pass

    if new_list is not no_update:
        _save_watchlist(new_list)

    return new_list


@app.callback(
    [Output("watchlist-chips", "children"),
     Output("selected-symbol", "options")],
    Input("watchlist-store", "data"),
)
def render_watchlist(symbols):
    if not symbols:
        return html.Span("No symbols — add tickers above or import from account.",
                         style={"color": COLORS["muted"], "fontSize": "13px"}), []

    chips = []
    for sym in symbols:
        chips.append(
            html.Span([
                html.Span(sym, style={"marginRight": "5px"}),
                html.Span(
                    "✕",
                    id={"type": "remove-sym-btn", "index": sym},
                    n_clicks=0,
                    style={
                        "cursor": "pointer",
                        "color": COLORS["muted"],
                        "fontSize": "11px",
                        "fontWeight": "700",
                        ":hover": {"color": COLORS["red"]},
                    },
                ),
            ], style={
                "display": "inline-flex",
                "alignItems": "center",
                "background": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "4px",
                "padding": "4px 10px",
                "marginRight": "6px",
                "marginBottom": "4px",
                "fontSize": "13px",
                "color": COLORS["text"],
                "fontWeight": "600",
            })
        )

    options = [{"label": s, "value": s} for s in symbols]
    return html.Div(chips, style={"display": "flex", "flexWrap": "wrap"}), options


@app.callback(
    [Output("quote-bar", "children"),
     Output("last-update", "children"),
     Output("positions-store", "data")],
    Input("auto-refresh", "n_intervals"),
    Input("watchlist-store", "data"),
    prevent_initial_call=False,
)
def refresh_quotes(_, symbols):
    if not symbols:
        return html.Span("", style={}), "", []

    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        quotes = client.get_quotes(symbols)
    except Exception as e:
        err = html.Span(f"⚠ Quote error: {e}", style={"color": COLORS["red"], "fontSize": "12px"})
        return err, f"Error at {timestamp}", []

    chips = []
    for sym in symbols:
        q = quotes.get(sym, {}).get("quote", {})
        last  = q.get("lastPrice", q.get("mark", 0))
        chg   = q.get("netChange", 0)
        pct   = q.get("netPercentChangeInDouble", 0)
        color = COLORS["green"] if chg >= 0 else COLORS["red"]
        arrow = "▲" if chg >= 0 else "▼"

        chips.append(html.Div([
            html.Span(sym,   style={"fontWeight": "700", "color": COLORS["text"],
                                     "marginRight": "6px", "fontSize": "13px"}),
            html.Span(f"${last:.2f}", style={"color": color, "fontWeight": "600",
                                              "fontSize": "13px"}),
            html.Span(f" {arrow}{abs(pct):.2f}%",
                      style={"color": color, "fontSize": "11px", "marginLeft": "4px"}),
        ], style={"display": "inline-block", "marginRight": "18px",
                  "padding": "4px 10px", "background": COLORS["card"],
                  "borderRadius": "4px", "border": f"1px solid {COLORS['border']}"}))

    # Also refresh positions
    positions = []
    try:
        positions = client.get_positions()
    except Exception:
        pass

    return html.Div(chips, style={"display": "flex", "flexWrap": "wrap"}), \
           f"Last refresh: {timestamp}", positions


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_monthly_expiry(available: list[str]) -> str | None:
    """
    Return the nearest upcoming standard monthly expiry (3rd Friday) from `available`.

    Four-pass strategy, each a safety net for the one above:

    Pass 1 — Exact 3rd-Friday match within ±2 days.
      Computes the true 3rd Friday for the current and next 5 months, then
      checks each date ±2 days against the available list.  The ±2 window
      handles: (a) index options listed as Saturday settlement, (b) holiday
      shifts that push expiry to Thursday or Monday, (c) symbols that round
      to a nearby date.  First hit wins — guaranteed to be within a week of
      the true monthly cycle.

    Pass 2 — "3rd-Friday zone" scan.
      Looks for any available date whose day-of-month falls in 15–21 AND is
      a Thursday–Saturday (±1 from Friday).  Symbols that only carry
      quarterly or semi-annual expirations fall here; the first future date
      in the right calendar zone is returned.

    Pass 3 — Nearest date in the 14-21 day range of any month.
      Broader zone (days 14–21) regardless of weekday, sorted closest to
      30 DTE.  Catches exchange-specific non-standard cycles.

    Pass 4 — Last resort: available date closest to 30 DTE (≥ 14 days out).
    """
    from datetime import date as _date, timedelta as _td

    today = _date.today()
    avail_set = set(available)

    def _third_friday(year: int, month: int) -> _date:
        first = _date(year, month, 1)
        days_to_first_fri = (4 - first.weekday()) % 7
        return first + _td(days=days_to_first_fri + 14)

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    year, month = today.year, today.month
    for _ in range(5):
        tf = _third_friday(year, month)
        if tf >= today:
            for delta in (0, 1, -1, 2, -2):   # Friday → Sat → Thu → Mon → Wed
                candidate = (tf + _td(days=delta)).isoformat()
                if candidate in avail_set:
                    return candidate
        month += 1
        if month > 12:
            month, year = 1, year + 1

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    candidates = []
    for e in available:
        try:
            d = _date.fromisoformat(e)
            if d >= today and 15 <= d.day <= 21 and d.weekday() in (3, 4, 5):
                candidates.append((d, e))
        except ValueError:
            continue
    if candidates:
        return sorted(candidates)[0][1]

    # ── Pass 3 ───────────────────────────────────────────────────────────────
    candidates = []
    for e in available:
        try:
            d = _date.fromisoformat(e)
            dte = (d - today).days
            if dte >= 0 and 14 <= d.day <= 21:
                candidates.append((abs(dte - 30), d, e))
        except ValueError:
            continue
    if candidates:
        return sorted(candidates)[0][2]

    # ── Pass 4 ───────────────────────────────────────────────────────────────
    future = []
    for e in available:
        try:
            d = _date.fromisoformat(e)
            dte = (d - today).days
            if dte >= 14:
                future.append((abs(dte - 30), d, e))
        except ValueError:
            continue
    if future:
        return sorted(future)[0][2]

    return available[0] if available else None


# ── Auto-select last symbol on page load ─────────────────────────────────────

@app.callback(
    Output("selected-symbol", "value"),
    Input("watchlist-store", "data"),
    State("last-symbol-store", "data"),
    prevent_initial_call=False,
)
def auto_select_symbol(watchlist, last_symbol):
    """On load: restore last used symbol; fallback to first in watchlist."""
    watchlist = watchlist or []
    if last_symbol and last_symbol in watchlist:
        return last_symbol
    if watchlist:
        return watchlist[0]
    return no_update


@app.callback(
    Output("last-symbol-store", "data"),
    Input("url", "pathname"),          # restore from disk on every page load
    Input("selected-symbol", "value"),
    prevent_initial_call=False,
)
def save_last_symbol(_, symbol):
    prop_id = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if "url" in prop_id or not prop_id:
        return _load_last_symbol()
    if symbol:
        _save_last_symbol(symbol)
    return symbol if symbol else no_update


def _build_chart_figure(symbol: str, candles: list) -> tuple:
    """Shared chart-building logic used by both callbacks below."""
    fig    = go.Figure()
    badges = []

    if candles:
        df    = compute_smas(candles)
        dates = df.index

        fig.add_trace(go.Candlestick(
            x=dates,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            name=symbol, showlegend=False,
        ))
        for col, color, label in [("sma_8",  COLORS["sma8"],  "SMA 8"),
                                   ("sma_20", COLORS["sma20"], "SMA 20"),
                                   ("sma_50", COLORS["sma50"], "SMA 50")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=dates, y=df[col], mode="lines", name=label,
                    line=dict(color=color, width=1.5),
                ))

        last_price = float(df["close"].iloc[-1])
        fig.add_hline(y=last_price, line_color=COLORS["blue"], line_dash="dash",
                      line_width=1, annotation_text=f"  ${last_price:.2f}",
                      annotation_font_color=COLORS["blue"])

        smas      = latest_smas(candles)
        signal    = sma_signal(last_price, smas)
        sig_color = {"BULLISH": COLORS["green"], "BEARISH": COLORS["red"]}.get(signal, COLORS["yellow"])

        for label, key, color in [("SMA 8",  "sma_8",  COLORS["sma8"]),
                                   ("SMA 20", "sma_20", COLORS["sma20"]),
                                   ("SMA 50", "sma_50", COLORS["sma50"])]:
            val      = smas.get(key)
            diff     = round(last_price - val, 2) if val else None
            diff_str = f" ({'+' if diff and diff >= 0 else ''}{diff})" if diff is not None else ""
            badges.append(html.Span(
                f"{label}: {'N/A' if val is None else f'${val:.2f}'}{diff_str}",
                style={"color": color, "marginRight": "16px",
                       "fontSize": "12px", "fontWeight": "600"},
            ))
        badges.append(html.Span(f"| Signal: {signal}",
                                style={"color": sig_color, "fontSize": "12px",
                                       "fontWeight": "700"}))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11),
        xaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                   rangeslider=dict(visible=False), type="date"),
        yaxis=dict(gridcolor=COLORS["border"], showgrid=True, side="right"),
        margin=dict(l=0, r=60, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )
    return fig, html.Div(badges)


# ── Callback 1: Chart figure + header — fires on symbol change OR auto-refresh
# Does NOT touch expiry selector, so the user's selection is preserved.
@app.callback(
    [Output("price-chart", "figure"),
     Output("sma-badges", "children"),
     Output("chart-symbol-header", "children")],
    Input("selected-symbol", "value"),
    Input("auto-refresh", "n_intervals"),
    prevent_initial_call=True,
)
def update_chart_figure(symbol, _):
    empty_fig = go.Figure(layout=dict(paper_bgcolor=COLORS["bg"],
                                       plot_bgcolor=COLORS["bg"]))
    if not symbol:
        return empty_fig, "", ""

    company_name = _get_company_name(symbol)

    candles = []
    try:
        hist    = client.get_price_history(symbol, period_type="month", period=3)
        candles = hist.get("candles", [])
    except Exception:
        pass

    fig, badges = _build_chart_figure(symbol, candles)

    header = html.Div([
        html.Span(symbol, style={"fontSize": "22px", "fontWeight": "900",
                                  "color": COLORS["blue"], "marginRight": "10px"}),
        html.Span(company_name if company_name != symbol else "",
                  style={"fontSize": "14px", "color": COLORS["muted"]}),
    ])
    return fig, badges, header


# ── Callback 2: Expiry list — fires ONLY when symbol changes (never on refresh)
# This is the only place that sets expiry-selector value.
@app.callback(
    [Output("expiry-selector", "options"),
     Output("expiry-selector", "value")],
    Input("selected-symbol", "value"),
    prevent_initial_call=True,
)
def init_expiry_options(symbol):
    if not symbol:
        return [], None

    expiry_options = []
    try:
        if hasattr(client, "get_expirations"):
            exps = client.get_expirations(symbol)
        else:
            raw  = client.get_option_chain(symbol, strike_count=1)
            cdf  = parse_option_chain(raw)
            exps = sorted(cdf[cdf["dte"] >= 0]["expiry"].unique().tolist()) if not cdf.empty else []

        today = datetime.today().date()
        for e in exps:
            try:
                dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
                expiry_options.append({"label": f"{e}  ({dte}d)", "value": e})
            except Exception:
                pass
    except Exception:
        pass

    if not expiry_options:
        return [], None

    avail = [o["value"] for o in expiry_options]
    print(f"[Expiry] {symbol} available: {avail[:10]}")
    default = _next_monthly_expiry(avail)
    print(f"[Expiry] {symbol} → monthly selected: {default}")
    return expiry_options, default


@app.callback(
    Output("chain-store", "data"),
    Input("expiry-selector", "value"),
    State("selected-symbol", "value"),
    prevent_initial_call=True,
)
def load_chain_for_expiry(expiry, symbol):
    """
    Fires when an expiry is chosen (including auto-selection of nearest).
    Fetches the full option chain for just that one expiry date — fast.
    """
    if not expiry or not symbol:
        return {}
    try:
        if hasattr(client, "get_option_chain"):
            # Pass expiry directly to yfinance client
            chain = client.get_option_chain(symbol, strike_count=40, expiry=expiry)
        else:
            chain = client.get_option_chain(symbol, strike_count=40)
        return chain
    except Exception:
        return {}


@app.callback(
    [Output("gamma-chart", "figure"),
     Output("gamma-table", "children"),
     Output("oi-chart", "figure"),
     Output("oi-table", "children"),
     Output("levels-summary", "children")],
    Input("chain-store", "data"),
    State("expiry-selector", "value"),
    State("selected-symbol", "value"),
    prevent_initial_call=True,
)
def update_options_panels(chain_data, expiry, symbol):
    empty_fig = go.Figure(layout=dict(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"]))

    if not chain_data or not symbol:
        return empty_fig, "", empty_fig, "", ""

    try:
        chain_df = parse_option_chain(chain_data)
    except Exception:
        return empty_fig, "", empty_fig, "", ""

    if chain_df.empty:
        return empty_fig, "", empty_fig, "", ""

    underlying_price = float(chain_data.get("underlyingPrice", 0))

    # ── Gamma chart ──────────────────────────────────────────────────────
    # Compute net GEX for ALL strikes (no z-threshold gate on the chart),
    # highlight outliers visually via opacity.
    df_exp = chain_df[chain_df["expiry"] == expiry].copy() if expiry else chain_df.copy()

    gex_all = pd.DataFrame()
    if not df_exp.empty:
        df_exp["gex"] = df_exp.apply(
            lambda r: r["gamma"] * r["open_interest"] * 100 * (1 if r["put_call"] == "CALL" else -1),
            axis=1,
        )
        gex_all = (
            df_exp.groupby("strike")["gex"]
            .sum()
            .reset_index()
            .rename(columns={"gex": "gex_net"})
            .sort_values("strike")
        )
        # Mark outliers
        if len(gex_all) > 2:
            mu  = gex_all["gex_net"].abs().mean()
            std = gex_all["gex_net"].abs().std()
            gex_all["z"] = (gex_all["gex_net"].abs() - mu) / std if std > 0 else 0
        else:
            gex_all["z"] = 0

    gamma_fig = go.Figure()
    if not gex_all.empty:
        bar_colors = [
            (COLORS["green"] if v > 0 else COLORS["red"])
            for v in gex_all["gex_net"]
        ]
        gamma_fig.add_trace(go.Bar(
            x=gex_all["strike"],
            y=gex_all["gex_net"],
            marker_color=bar_colors,
            name="Net GEX",
            hovertemplate="Strike: %{x}<br>Net GEX: %{y:,.0f}<extra></extra>",
        ))
        if underlying_price:
            gamma_fig.add_vline(x=underlying_price, line_color=COLORS["blue"],
                                line_dash="dash", line_width=2,
                                annotation_text=f"  ${underlying_price:.2f}",
                                annotation_font_color=COLORS["blue"])

    gamma_fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11),
        xaxis=dict(gridcolor=COLORS["border"], title="Strike"),
        yaxis=dict(gridcolor=COLORS["border"], title="Net GEX"),
        margin=dict(l=40, r=10, t=10, b=40),
        showlegend=False,
    )

    # Table: top 10 by absolute GEX
    gamma_table_rows = []
    if not gex_all.empty:
        top_gex = gex_all.reindex(gex_all["gex_net"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top_gex.iterrows():
            gex_color = COLORS["green"] if r["gex_net"] > 0 else COLORS["red"]
            label = "CALL WALL" if r["gex_net"] > 0 else "PUT WALL"
            outlier = " ⚡" if r.get("z", 0) >= 1.2 else ""
            gamma_table_rows.append(html.Tr([
                html.Td(f"${r['strike']:.1f}", style={"color": COLORS["text"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{r['gex_net']:+,.0f}", style={"color": gex_color, "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{r.get('z', 0):.1f}σ{outlier}", style={"color": COLORS["muted"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(label, style={"color": gex_color, "fontSize": "11px", "padding": "3px 6px"}),
            ]))

    gamma_tbl = html.Table([
        html.Thead(html.Tr([
            html.Th("Strike", style=_th()), html.Th("Net GEX", style=_th()),
            html.Th("Z-Score", style=_th()), html.Th("Type", style=_th()),
        ])),
        html.Tbody(gamma_table_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"}) if gamma_table_rows else html.Span(
        "No options data — select a symbol and expiry above.", style={"color": COLORS["muted"], "fontSize": "12px"})

    # ── OI chart — show all strikes sorted by strike, top 20 by total OI ──
    oi_fig = go.Figure()
    oi_table_rows = []

    if not df_exp.empty:
        oi_by_strike = (
            df_exp.groupby(["strike", "put_call"])["open_interest"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["CALL", "PUT"]:
            if col not in oi_by_strike.columns:
                oi_by_strike[col] = 0

        oi_by_strike["total_oi"] = oi_by_strike["CALL"] + oi_by_strike["PUT"]
        oi_by_strike["pcr"] = oi_by_strike.apply(
            lambda r: round(r["PUT"] / r["CALL"], 2) if r["CALL"] > 0 else 999, axis=1
        )
        if len(oi_by_strike) > 2:
            mu  = oi_by_strike["total_oi"].mean()
            std = oi_by_strike["total_oi"].std()
            oi_by_strike["z"] = (oi_by_strike["total_oi"] - mu) / std if std > 0 else 0
        else:
            oi_by_strike["z"] = 0

        oi_chart = oi_by_strike.sort_values("strike")

        oi_fig.add_trace(go.Bar(
            x=oi_chart["strike"], y=oi_chart["CALL"],
            name="Call OI", marker_color=COLORS["green"],
            hovertemplate="Strike: %{x}<br>Call OI: %{y:,}<extra></extra>",
        ))
        oi_fig.add_trace(go.Bar(
            x=oi_chart["strike"], y=-oi_chart["PUT"],
            name="Put OI", marker_color=COLORS["red"],
            hovertemplate="Strike: %{x}<br>Put OI: %{y:,}<extra></extra>",
        ))
        if underlying_price:
            oi_fig.add_vline(x=underlying_price, line_color=COLORS["blue"],
                             line_dash="dash", line_width=2)

        # Table: top 15 by total OI
        top_oi = oi_by_strike.sort_values("total_oi", ascending=False).head(15)
        for _, r in top_oi.iterrows():
            pcr = r["pcr"]
            bias = "PUT HEAVY" if pcr > 1.5 else ("CALL HEAVY" if pcr < 0.67 else "BALANCED")
            bias_color = COLORS["red"] if pcr > 1.5 else (COLORS["green"] if pcr < 0.67 else COLORS["muted"])
            outlier = " ⚡" if r.get("z", 0) >= 1.2 else ""
            oi_table_rows.append(html.Tr([
                html.Td(f"${r['strike']:.1f}", style={"color": COLORS["text"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{int(r['CALL']):,}", style={"color": COLORS["green"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{int(r['PUT']):,}", style={"color": COLORS["red"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{pcr:.2f}", style={"color": COLORS["muted"], "fontSize": "12px", "padding": "3px 6px"}),
                html.Td(f"{bias}{outlier}", style={"color": bias_color, "fontSize": "11px", "padding": "3px 6px"}),
            ]))

    oi_fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=11),
        xaxis=dict(gridcolor=COLORS["border"], title="Strike"),
        yaxis=dict(gridcolor=COLORS["border"], title="Open Interest"),
        margin=dict(l=40, r=10, t=10, b=40),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )

    oi_tbl = html.Table([
        html.Thead(html.Tr([
            html.Th("Strike", style=_th()), html.Th("Call OI", style=_th()),
            html.Th("Put OI", style=_th()), html.Th("P/C Ratio", style=_th()),
            html.Th("Bias", style=_th()),
        ])),
        html.Tbody(oi_table_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"}) if oi_table_rows else html.Span(
        "No options data — select a symbol and expiry above.", style={"color": COLORS["muted"], "fontSize": "12px"})

    # ── Levels summary ────────────────────────────────────────────────────
    levels = get_key_levels(chain_df, underlying_price, expiry)
    resistance = levels["nearest_resistance"]
    support    = levels["nearest_support"]
    max_pain   = levels["max_pain"]

    dist_res = round(resistance - underlying_price, 2) if resistance else None
    dist_sup = round(underlying_price - support, 2) if support else None

    summary = html.Div([
        _level_row("Current Price",  f"${underlying_price:.2f}", COLORS["blue"]),
        _level_row("Nearest Resistance (Call Wall)",
                   f"${resistance:.0f}  (+{dist_res:.2f})" if resistance else "N/A", COLORS["green"]),
        _level_row("Nearest Support (Put Wall)",
                   f"${support:.0f}  (-{dist_sup:.2f})" if support else "N/A", COLORS["red"]),
        _level_row("Max Pain Strike", f"${max_pain:.0f}" if max_pain else "N/A", COLORS["yellow"]),
        html.Hr(style={"border": f"1px solid {COLORS['border']}", "margin": "10px 0"}),
        html.Div([
            html.Span("Selected Expiry: ", style={"color": COLORS["muted"], "fontSize": "12px"}),
            html.Span(expiry or "All", style={"color": COLORS["text"], "fontSize": "12px", "fontWeight": "700"}),
        ]),
    ])
    return gamma_fig, gamma_tbl, oi_fig, oi_tbl, summary


@app.callback(
    Output("positions-table", "children"),
    Input("positions-store", "data"),
    Input("chain-store", "data"),
)
def render_positions(positions, chain_data):
    if not positions:
        if DATA_SOURCE == "yfinance":
            return html.Div([
                html.Span("⏳ Waiting for Schwab API approval.", style={"color": COLORS["yellow"], "fontSize": "13px"}),
                html.Br(),
                html.Span("Positions will appear here once connected.", style={"color": COLORS["muted"], "fontSize": "12px"}),
            ])
        return html.Span("No open positions.", style={"color": COLORS["muted"], "fontSize": "13px"})

    rows = []
    for p in positions:
        pnl = p.get("unrealized_pnl", 0)
        pnl_color = COLORS["green"] if pnl >= 0 else COLORS["red"]
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

        sym = p.get("underlying", p.get("symbol", ""))
        detail = ""
        if p.get("asset_type") == "OPTION":
            detail = f"{p.get('put_call','')} ${p.get('strike','')} {p.get('expiry','')[:10]}"

        rows.append(html.Tr([
            html.Td(sym, style={"color": COLORS["blue"], "fontSize": "12px",
                                 "fontWeight": "700", "paddingRight": "8px"}),
            html.Td(detail or p.get("asset_type",""), style={"color": COLORS["muted"], "fontSize": "11px"}),
            html.Td(f"{p.get('quantity',0):+.0f}", style={"color": COLORS["text"], "fontSize": "12px"}),
            html.Td(pnl_str, style={"color": pnl_color, "fontSize": "12px", "fontWeight": "600"}),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Symbol", style=_th()), html.Th("Details", style=_th()),
            html.Th("Qty", style=_th()),   html.Th("Day P&L", style=_th()),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


@app.callback(
    Output("earnings-table", "children"),
    Input("watchlist-store", "data"),
    Input("auto-refresh", "n_intervals"),
)
def render_earnings(symbols, _):
    if not symbols:
        return html.Span("No symbols loaded.", style={"color": COLORS["muted"]})

    earnings = get_earnings_dates(symbols)
    rows = []
    for sym, info in sorted(earnings.items(), key=lambda x: (x[1].get("days_away") or 9999)):
        days = info.get("days_away")
        color = (COLORS["red"] if days is not None and days < 7
                 else COLORS["yellow"] if days is not None and days < 14
                 else COLORS["text"])
        rows.append(html.Tr([
            html.Td(sym, style={"color": COLORS["blue"], "fontWeight": "700", "fontSize": "13px"}),
            html.Td(info.get("date", "N/A"), style={"color": color, "fontSize": "12px"}),
            html.Td(f"{days}d" if days is not None and days >= 0 else "—",
                    style={"color": color, "fontSize": "12px"}),
            html.Td(info.get("flag", ""), style={"fontSize": "12px"}),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Symbol", style=_th()), html.Th("Earnings Date", style=_th()),
            html.Th("Days Away", style=_th()), html.Th("Alert", style=_th()),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _th():
    return {"color": COLORS["muted"], "fontSize": "11px", "fontWeight": "600",
            "borderBottom": f"1px solid {COLORS['border']}", "padding": "4px 6px",
            "textAlign": "left", "letterSpacing": "0.5px"}


def _level_row(label: str, value: str, color: str):
    return html.Div([
        html.Span(label + ": ", style={"color": COLORS["muted"], "fontSize": "12px",
                                        "display": "inline-block", "width": "220px"}),
        html.Span(value, style={"color": color, "fontWeight": "700", "fontSize": "13px"}),
    ], style={"marginBottom": "8px"})


@app.callback(
    Output("sma-watchlist-table", "children"),
    Input("watchlist-store", "data"),
    Input("auto-refresh", "n_intervals"),
)
def render_sma_watchlist(symbols, _):
    if not symbols:
        return html.Span("Add symbols above to see SMA levels.",
                         style={"color": COLORS["muted"], "fontSize": "13px"})

    rows = []
    for sym in symbols:
        try:
            hist = client.get_price_history(sym, period_type="month", period=3)
            candles = hist.get("candles", [])
            if not candles:
                raise ValueError("no candles")
            df = compute_smas(candles)
            last   = round(float(df["close"].iloc[-1]), 2)
            smas   = latest_smas(candles)
            signal = sma_signal(last, smas)
            signal_color = {"BULLISH": COLORS["green"], "BEARISH": COLORS["red"]}.get(signal, COLORS["yellow"])

            def _sma_cell(key):
                val = smas.get(key)
                if val is None:
                    return html.Td("—", style={"color": COLORS["muted"], "fontSize": "12px", "padding": "5px 10px"})
                diff = last - val
                color = COLORS["green"] if diff > 0 else COLORS["red"]
                return html.Td([
                    html.Span(f"${val:.2f}", style={"color": COLORS["text"], "fontSize": "12px"}),
                    html.Span(f" ({'+' if diff>=0 else ''}{diff:.2f})",
                              style={"color": color, "fontSize": "11px"}),
                ], style={"padding": "5px 10px"})

            rows.append(html.Tr([
                html.Td(sym, style={
                    "color": COLORS["blue"], "fontWeight": "700",
                    "fontSize": "13px", "padding": "5px 10px"
                }),
                html.Td(f"${last:.2f}", style={
                    "color": COLORS["text"], "fontWeight": "600",
                    "fontSize": "13px", "padding": "5px 10px"
                }),
                _sma_cell("sma_8"),
                _sma_cell("sma_20"),
                _sma_cell("sma_50"),
                html.Td(signal, style={
                    "color": signal_color, "fontWeight": "700",
                    "fontSize": "12px", "padding": "5px 10px"
                }),
            ]))
        except Exception:
            rows.append(html.Tr([
                html.Td(sym, style={"color": COLORS["blue"], "fontWeight": "700",
                                     "fontSize": "13px", "padding": "5px 10px"}),
                html.Td("—", colSpan=5, style={"color": COLORS["muted"],
                                                "fontSize": "12px", "padding": "5px 10px"}),
            ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Symbol",  style=_th()),
            html.Th("Price",   style=_th()),
            html.Th("SMA 8",   style={**_th(), "color": COLORS["sma8"]}),
            html.Th("SMA 20",  style={**_th(), "color": COLORS["sma20"]}),
            html.Th("SMA 50",  style={**_th(), "color": COLORS["sma50"]}),
            html.Th("Signal",  style=_th()),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


@app.callback(
    Output("gamma-watchlist-table", "children"),
    Input("watchlist-store", "data"),
    Input("auto-refresh", "n_intervals"),
)
def render_gamma_watchlist(symbols, _):
    """
    For each symbol: fetch its next monthly expiry chain, find the largest
    call wall and put wall by net GEX, and display in a summary table.
    """
    if not symbols:
        return html.Span("Add symbols above to see gamma walls.",
                         style={"color": COLORS["muted"], "fontSize": "13px"})

    rows = []
    for i, sym in enumerate(symbols):

        try:
            # 1. Get available expiries and find the next monthly
            if hasattr(client, "get_expirations"):
                exps = client.get_expirations(sym)
            else:
                raw = client.get_option_chain(sym, strike_count=1)
                cdf = parse_option_chain(raw)
                exps = sorted(cdf[cdf["dte"] >= 0]["expiry"].unique().tolist()) if not cdf.empty else []

            # Debug: print to Terminal so we can see what yfinance returns
            print(f"[GEX table] {sym} expiries: {exps[:8]}")

            monthly = _next_monthly_expiry(exps)
            print(f"[GEX table] {sym} → selected monthly: {monthly}")
            if not monthly:
                raise ValueError("no monthly expiry found")

            # 2. Fetch chain for that expiry
            if hasattr(client, "get_option_chain"):
                chain_data = client.get_option_chain(sym, strike_count=40, expiry=monthly)
            else:
                chain_data = client.get_option_chain(sym, strike_count=40)

            chain_df = parse_option_chain(chain_data)
            if chain_df.empty:
                raise ValueError("empty chain")

            spot = float(chain_data.get("underlyingPrice", 0))

            # 3. Compute net GEX per strike for this expiry
            df_m = chain_df[chain_df["expiry"] == monthly].copy()
            if df_m.empty:
                # expiry key might include DTE suffix — match on date portion
                df_m = chain_df[chain_df["expiry"].str.startswith(monthly[:10])].copy()

            if df_m.empty:
                raise ValueError("no data for monthly expiry")

            df_m["gex"] = df_m.apply(
                lambda r: r["gamma"] * r["open_interest"] * 100
                          * (1 if r["put_call"] == "CALL" else -1),
                axis=1,
            )
            gex_by_strike = (
                df_m.groupby("strike")["gex"]
                .sum()
                .reset_index()
                .rename(columns={"gex": "gex_net"})
            )

            # Largest call wall (most positive GEX above spot)
            calls_above = gex_by_strike[
                (gex_by_strike["gex_net"] > 0) & (gex_by_strike["strike"] > spot)
            ]
            top_call = calls_above.loc[calls_above["gex_net"].idxmax()] \
                if not calls_above.empty else None

            # Largest put wall (most negative GEX below spot)
            puts_below = gex_by_strike[
                (gex_by_strike["gex_net"] < 0) & (gex_by_strike["strike"] < spot)
            ]
            top_put = puts_below.loc[puts_below["gex_net"].idxmin()] \
                if not puts_below.empty else None

            # Net bias
            total_pos = gex_by_strike[gex_by_strike["gex_net"] > 0]["gex_net"].sum()
            total_neg = gex_by_strike[gex_by_strike["gex_net"] < 0]["gex_net"].sum()
            net = total_pos + total_neg
            bias = "CALL HEAVY" if net > 0 else "PUT HEAVY"
            bias_color = COLORS["green"] if net > 0 else COLORS["red"]

            def _fmt_wall(row, color):
                if row is None:
                    return html.Td("—", style={"color": COLORS["muted"],
                                               "fontSize": "12px", "padding": "5px 10px"})
                dist = round(abs(row["strike"] - spot), 2)
                return html.Td([
                    html.Span(f"${row['strike']:.0f}",
                              style={"color": color, "fontWeight": "700", "fontSize": "13px"}),
                    html.Span(f"  GEX {row['gex_net']:+,.0f}",
                              style={"color": COLORS["muted"], "fontSize": "11px",
                                     "marginLeft": "6px"}),
                    html.Span(f"  ({dist:.2f} away)",
                              style={"color": COLORS["muted"], "fontSize": "10px",
                                     "marginLeft": "4px"}),
                ], style={"padding": "5px 10px"})

            rows.append(html.Tr([
                html.Td(sym, style={"color": COLORS["blue"], "fontWeight": "700",
                                     "fontSize": "13px", "padding": "5px 10px"}),
                html.Td(f"${spot:.2f}", style={"color": COLORS["text"], "fontWeight": "600",
                                                "fontSize": "13px", "padding": "5px 10px"}),
                html.Td(monthly, style={"color": COLORS["muted"], "fontSize": "12px",
                                         "padding": "5px 10px"}),
                _fmt_wall(top_call, COLORS["green"]),
                _fmt_wall(top_put,  COLORS["red"]),
                html.Td(bias, style={"color": bias_color, "fontWeight": "700",
                                      "fontSize": "12px", "padding": "5px 10px"}),
            ]))

        except Exception as e:
            err_msg = str(e) or type(e).__name__
            print(f"[GEX table] {sym} ERROR: {err_msg}")
            rows.append(html.Tr([
                html.Td(sym, style={"color": COLORS["blue"], "fontWeight": "700",
                                     "fontSize": "13px", "padding": "5px 10px"}),
                html.Td(f"Error: {err_msg}", colSpan=5,
                        style={"color": COLORS["red"], "fontSize": "11px",
                               "padding": "5px 10px"}),
            ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Symbol",     style=_th()),
            html.Th("Price",      style=_th()),
            html.Th("Monthly Exp", style=_th()),
            html.Th("Call Wall",  style={**_th(), "color": COLORS["green"]}),
            html.Th("Put Wall",   style={**_th(), "color": COLORS["red"]}),
            html.Th("GEX Bias",   style=_th()),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Starting Derivatives Dashboard → http://127.0.0.1:8050\n")
    port = int(os.getenv("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
