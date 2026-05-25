"""
Schwab API Client — OAuth2 + Market Data + Account Access
Handles token lifecycle, auto-refresh, and all API calls needed
by the derivatives dashboard. TD Ameritrade / ThinkorSwim accounts
are now served through the same Schwab API.

Register your app at: https://developer.schwab.com/
Redirect URI must be set to: https://127.0.0.1
"""

import os
import json
import time
import base64
import threading
import webbrowser
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Schwab API base URLs ────────────────────────────────────────────────────
AUTH_URL     = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"
MARKET_BASE  = "https://api.schwabapi.com/marketdata/v1"
TRADER_BASE  = "https://api.schwabapi.com/trader/v1"
TOKEN_FILE   = Path(".schwab_tokens.json")


# ── Local OAuth callback server ─────────────────────────────────────────────
class _CallbackHandler(BaseHTTPRequestHandler):
    auth_code = None

    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>Auth complete. You can close this tab.</h2>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h2>Auth failed. No code found.</h2>")

    def log_message(self, *_):
        pass  # suppress access log


def _get_auth_code(app_key: str, redirect_uri: str) -> str:
    """Open browser for user login, capture auth code via local server."""
    params = {
        "response_type": "code",
        "client_id": app_key,
        "redirect_uri": redirect_uri,
        "scope": "readonly",
    }
    url = AUTH_URL + "?" + urllib.parse.urlencode(params)

    # Parse port from redirect_uri (default 443 → use 80 locally)
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 443

    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    _CallbackHandler.auth_code = None

    print(f"\n🔐  Opening browser for Schwab login…\n    {url}\n")
    webbrowser.open(url)

    server.handle_request()
    if not _CallbackHandler.auth_code:
        raise RuntimeError("Auth code not captured — did you complete login?")
    return _CallbackHandler.auth_code


# ── Main client ─────────────────────────────────────────────────────────────
class SchwabClient:
    """
    Usage
    -----
    client = SchwabClient()
    client.authenticate()          # first run opens browser
    quotes = client.get_quotes(["SPY", "QQQ"])
    chain  = client.get_option_chain("SPY")
    hist   = client.get_price_history("SPY", period_type="month", period=3)
    positions = client.get_positions()
    """

    def __init__(self):
        self.app_key     = os.getenv("SCHWAB_APP_KEY", "")
        self.app_secret  = os.getenv("SCHWAB_APP_SECRET", "")
        self.redirect    = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
        self._access_token  = None
        self._refresh_token = None
        self._token_expiry  = 0
        self._lock = threading.Lock()

        if not self.app_key or not self.app_secret:
            raise EnvironmentError(
                "Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET in your .env file.\n"
                "Register at https://developer.schwab.com/"
            )

        self._load_tokens()

    # ── Token persistence ──────────────────────────────────────────────────
    def _load_tokens(self):
        if TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text())
                self._access_token  = data.get("access_token")
                self._refresh_token = data.get("refresh_token")
                self._token_expiry  = data.get("expiry", 0)
            except Exception:
                pass

    def _save_tokens(self):
        TOKEN_FILE.write_text(json.dumps({
            "access_token":  self._access_token,
            "refresh_token": self._refresh_token,
            "expiry":        self._token_expiry,
        }))

    # ── Auth helpers ───────────────────────────────────────────────────────
    def _basic_auth(self) -> str:
        creds = f"{self.app_key}:{self.app_secret}"
        return "Basic " + base64.b64encode(creds.encode()).decode()

    def _exchange_code(self, code: str):
        r = requests.post(TOKEN_URL, headers={
            "Authorization": self._basic_auth(),
            "Content-Type":  "application/x-www-form-urlencoded",
        }, data={
            "grant_type":   "authorization_code",
            "code":          code,
            "redirect_uri":  self.redirect,
        })
        r.raise_for_status()
        self._store_token_response(r.json())

    def _refresh_access_token(self):
        if not self._refresh_token:
            raise RuntimeError("No refresh token — call authenticate() first.")
        r = requests.post(TOKEN_URL, headers={
            "Authorization": self._basic_auth(),
            "Content-Type":  "application/x-www-form-urlencoded",
        }, data={
            "grant_type":    "refresh_token",
            "refresh_token":  self._refresh_token,
        })
        r.raise_for_status()
        self._store_token_response(r.json())

    def _store_token_response(self, data: dict):
        self._access_token  = data["access_token"]
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        expires_in          = data.get("expires_in", 1800)
        self._token_expiry  = time.time() + expires_in - 60  # 60s buffer
        self._save_tokens()

    def authenticate(self):
        """Full OAuth2 flow: open browser, capture code, exchange for tokens."""
        code = _get_auth_code(self.app_key, self.redirect)
        self._exchange_code(code)
        print("✅  Authenticated with Schwab API.")

    def _ensure_token(self):
        with self._lock:
            if time.time() >= self._token_expiry:
                if self._refresh_token:
                    try:
                        self._refresh_access_token()
                    except Exception:
                        raise RuntimeError(
                            "Token refresh failed. Run authenticate() again."
                        )
                else:
                    raise RuntimeError("Not authenticated. Call authenticate() first.")

    def _headers(self) -> dict:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def is_authenticated(self) -> bool:
        return bool(self._access_token and self._refresh_token)

    # ── Market Data ────────────────────────────────────────────────────────
    def get_quotes(self, symbols: list[str]) -> dict:
        """
        Returns real-time quotes for a list of symbols.
        Each value includes: last, bid, ask, volume, change, %change, etc.
        """
        r = requests.get(f"{MARKET_BASE}/quotes", headers=self._headers(), params={
            "symbols": ",".join(s.upper() for s in symbols),
            "fields":  "quote,reference",
            "indicative": "false",
        })
        r.raise_for_status()
        return r.json()

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 3,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> dict:
        """
        OHLCV history for SMA calculation.
        period_type: day | month | year | ytd
        frequency_type: minute | daily | weekly | monthly
        """
        r = requests.get(f"{MARKET_BASE}/pricehistory", headers=self._headers(), params={
            "symbol":        symbol.upper(),
            "periodType":    period_type,
            "period":        period,
            "frequencyType": frequency_type,
            "frequency":     frequency,
            "needExtendedHoursData": "false",
        })
        r.raise_for_status()
        return r.json()

    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "ALL",
        strike_count: int = 30,
        include_underlying: bool = True,
        strategy: str = "SINGLE",
        from_date: str | None = None,
        to_date: str | None = None,
        expiry: str | None = None,   # narrows request to a single expiry date
    ) -> dict:
        """
        Full options chain with OI, volume, and greeks (delta, gamma, theta, vega, rho).
        strike_count: number of strikes on each side of ATM
        """
        today = datetime.today()
        # If a single expiry is requested, pin both date bounds to that day
        if expiry:
            from_date = expiry
            to_date   = expiry
        params = {
            "symbol":            symbol.upper(),
            "contractType":      contract_type,
            "strikeCount":       strike_count,
            "includeUnderlyingQuote": str(include_underlying).lower(),
            "strategy":          strategy,
            "fromDate":          from_date or today.strftime("%Y-%m-%d"),
            "toDate":            to_date   or (today + timedelta(days=120)).strftime("%Y-%m-%d"),
        }
        r = requests.get(f"{MARKET_BASE}/chains", headers=self._headers(), params=params)
        r.raise_for_status()
        return r.json()

    def get_expirations(self, symbol: str) -> list[str]:
        """
        Return sorted list of upcoming expiry date strings "YYYY-MM-DD" for a symbol.
        Fetches a minimal chain (strike_count=1) spanning 12 months to extract
        all available expiry dates without pulling full strike data.
        """
        today = datetime.today()
        try:
            raw = self.get_option_chain(
                symbol,
                strike_count=1,
                to_date=(today + timedelta(days=365)).strftime("%Y-%m-%d"),
            )
            from analytics import parse_option_chain
            cdf = parse_option_chain(raw)
            if cdf.empty:
                return []
            return sorted(cdf[cdf["dte"] >= 0]["expiry"].unique().tolist())
        except Exception:
            return []

    def get_market_hours(self, market: str = "option") -> dict:
        """Check if market is currently open."""
        today = datetime.today().strftime("%Y-%m-%d")
        r = requests.get(f"{MARKET_BASE}/markets", headers=self._headers(), params={
            "markets": market,
            "date":    today,
        })
        r.raise_for_status()
        return r.json()

    # ── Account / Positions ────────────────────────────────────────────────
    def get_accounts(self) -> list[dict]:
        """Return all linked accounts (Schwab brokerage + ThinkorSwim)."""
        r = requests.get(f"{TRADER_BASE}/accounts", headers=self._headers(), params={
            "fields": "positions",
        })
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> list[dict]:
        """
        Flatten all positions across all accounts into a unified list.
        Each entry: symbol, description, quantity, average_price,
                    market_value, unrealized_pnl, asset_type, put_call, expiry, strike
        """
        accounts = self.get_accounts()
        positions = []
        for acct in accounts:
            acct_num = acct.get("securitiesAccount", {}).get("accountNumber", "?")
            acct_type = acct.get("securitiesAccount", {}).get("type", "?")
            for pos in acct.get("securitiesAccount", {}).get("positions", []):
                instr = pos.get("instrument", {})
                asset_type = instr.get("assetType", "")
                entry = {
                    "account":        acct_num[-4:],   # last 4 digits
                    "account_type":   acct_type,
                    "symbol":         instr.get("symbol", ""),
                    "description":    instr.get("description", ""),
                    "asset_type":     asset_type,
                    "quantity":       pos.get("longQuantity", 0) - pos.get("shortQuantity", 0),
                    "average_price":  pos.get("averagePrice", 0),
                    "market_value":   pos.get("marketValue", 0),
                    "unrealized_pnl": pos.get("currentDayProfitLoss", 0),
                    "unrealized_pnl_pct": pos.get("currentDayProfitLossPercentage", 0),
                }
                # Options-specific fields
                if asset_type == "OPTION":
                    opt = instr.get("optionDeliverable", [{}])[0]
                    entry.update({
                        "put_call":    instr.get("putCall", ""),
                        "strike":      instr.get("strikePrice", 0),
                        "expiry":      instr.get("expirationDate", ""),
                        "underlying":  instr.get("underlyingSymbol", ""),
                        "multiplier":  instr.get("multiplier", 100),
                    })
                positions.append(entry)
        return positions

    def get_unique_underlyings(self) -> list[str]:
        """Extract unique underlying symbols from all open option positions."""
        positions = self.get_positions()
        symbols = set()
        for p in positions:
            if p["asset_type"] == "OPTION":
                symbols.add(p.get("underlying", p["symbol"]))
            else:
                symbols.add(p["symbol"])
        return sorted(symbols)
