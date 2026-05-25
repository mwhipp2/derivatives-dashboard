# Derivatives Trading Dashboard — Setup Guide

## What this is

A live derivatives dashboard running locally in your browser. It connects to your
Schwab brokerage (and ThinkorSwim — they share the same API now) to show:

- **Real-time quotes** for any symbols you track
- **Price chart** with SMA 8 / 20 / 50 overlay + directional signal
- **Gamma walls** — strikes where net GEX is a statistical outlier (dealer hedging pressure)
- **OI outliers** — strikes with anomalous open interest (pinning / magnet levels)
- **Max pain** calculation per expiry
- **Earnings calendar** with proximity alerts
- **Open positions** with live P&L pulled from your account

---

## Step 1 — Register on Schwab Developer Portal

1. Go to [https://developer.schwab.com/](https://developer.schwab.com/)
2. Log in with your **Schwab brokerage account** (same login as schwab.com)
3. Click **"My Apps" → "Create App"**
4. Fill in:
   - **App Name**: anything (e.g. "My Trading Dashboard")
   - **Callback URL**: `https://127.0.0.1`  ← exact, no trailing slash
   - **Products**: check **"Accounts and Trading Production"** + **"Market Data Production"**
5. Submit — approval is usually instant for individual accounts
6. Copy your **App Key** and **App Secret**

> **ThinkorSwim note**: If your TOS account is linked to your Schwab account (which it
> is for all accounts since the TD Ameritrade migration), positions and data from
> both accounts will appear automatically. No separate API needed.

---

## Step 2 — Configure credentials

```bash
# In the dashboard folder:
cp .env.example .env
```

Open `.env` and paste in your keys:

```
SCHWAB_APP_KEY=your_actual_key
SCHWAB_APP_SECRET=your_actual_secret
SCHWAB_REDIRECT_URI=https://127.0.0.1
```

---

## Step 3 — Install dependencies

Requires Python 3.11+

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run the dashboard

```bash
python app.py
```

**First run:** A browser window will open for Schwab login. Log in, approve the
permissions, and you'll see "Auth complete." — tokens are saved locally in
`.schwab_tokens.json` and auto-refreshed so you won't need to log in again.

**Subsequent runs:** Opens directly to `http://127.0.0.1:8050`

---

## Using the Dashboard

| Action | How |
|--------|-----|
| Add a symbol | Type ticker in the input field → click **＋ Add** |
| Import your positions | Click **⟳ Refresh Account Positions** |
| Analyze a symbol | Select it from the dropdown on the right |
| Change expiry | Use the **Select expiry** dropdown in the Gamma Walls panel |
| Data refresh | Automatic every 30 seconds; Schwab API rate limit is ~120 req/min |

### Reading the Gamma Wall chart

- **Green bars above zero** = Call walls — dealers are long gamma here, strong
  resistance as they sell into rallies
- **Red bars below zero** = Put walls — dealers short gamma here, strong support
  as they buy dips to delta-hedge
- Outlier threshold: **1.2σ** above mean absolute GEX (configurable in `analytics.py`)

### Reading the OI chart

- Stacked Call OI (green) / Put OI (red) by strike
- High total OI strikes act as **price magnets** into expiry (max pain effect)
- P/C ratio shown per strike — <0.67 = call-heavy, >1.5 = put-heavy

---

## Troubleshooting

**"Token refresh failed"** — Run `python -c "from schwab_client import SchwabClient; SchwabClient().authenticate()"` to re-authenticate.

**No positions showing** — Make sure your Schwab app has "Accounts and Trading Production" enabled. It can take a few minutes after approval.

**Options chain empty** — Schwab requires market hours for live chain data on some symbols. Try during regular trading hours (9:30–16:00 ET).

**Port conflict** — Change the port in `app.py`: `app.run(port=8051)`

---

## File structure

```
derivatives-dashboard/
├── app.py              ← Main Dash application (run this)
├── schwab_client.py    ← Schwab OAuth2 + all API calls
├── analytics.py        ← SMA, gamma walls, OI outliers, max pain
├── requirements.txt    ← Python dependencies
├── .env                ← Your API keys (never commit this)
├── .env.example        ← Template
└── .schwab_tokens.json ← Auto-created on first login (keep private)
```
