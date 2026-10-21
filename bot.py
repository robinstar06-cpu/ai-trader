#!/usr/bin/env python3
"""
AI Crypto Trader - experiment met demogeld (Alpaca Paper).

Elke run:
  1. koersen + indicatoren ophalen voor alle crypto/USD-munten op Alpaca (+ Fear & Greed Index)
  2. Kronos (voorspelmodel voor candles) laten voorspellen hoe elke munt de komende uren beweegt
  3. je demo-portefeuille ophalen en de voortgang richting het doel bijhouden
  4. eens per dag: de AI evalueert zichzelf en werkt zijn lessen bij
  5. de AI (NVIDIA NIM, standaard Kimi K3) vragen om een doelverdeling, met zijn plan,
     lessen, de Kronos-voorspelling en de uitkomst van zijn eerdere beslissingen
  6. veiligheidsregels toepassen en orders plaatsen op het PAPER-account
  7. logboek + statuspagina (README.md) bijwerken

bot.py zelf gebruikt alleen standaard Python 3.10+. Kronos heeft PyTorch nodig (requirements.txt);
ontbreekt dat, dan draait de bot gewoon zonder Kronos.
"""
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Amsterdam")
except Exception:  # pragma: no cover
    TZ = timezone.utc

# --------------------------------------------------------------------------
# Vaste adressen. De bot praat ALLEEN met het paper-adres: echt geld kan niet.
# --------------------------------------------------------------------------
PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/us"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATE_FILE = DATA_DIR / "state.json"
EQUITY_CSV = DATA_DIR / "equity.csv"
DECISIONS_CSV = DATA_DIR / "decisions.csv"
README = ROOT / "README.md"

STABLECOINS = {"USDT", "USDC", "DAI", "PYUSD", "USDG", "TUSD", "BUSD", "USDP", "FDUSD", "EURC", "RLUSD", "USDE"}


def env(name, default):
    """Leest een instelling; lege waarde = standaard."""
    val = os.getenv(name)
    return default if val is None or str(val).strip() == "" else str(val).strip()


# ---------------------------- Instellingen ---------------------------------
MODELS = [m.strip() for m in env("MODELS", "moonshotai/kimi-k3,z-ai/glm-5.3,deepseek-ai/deepseek-v4.1-flash").split(",") if m.strip()]
SYMBOLS = env("SYMBOLS", "ALL")                      # "ALL" of bv. "BTC/USD,ETH/USD"
MAX_PER_COIN_PCT = float(env("MAX_PER_COIN_PCT", "60"))
MIN_CASH_PCT = float(env("MIN_CASH_PCT", "0"))
REBALANCE_BAND_PCT = float(env("REBALANCE_BAND_PCT", "1"))   # kleinere wijzigingen negeren
MIN_TRADE_USD = float(env("MIN_TRADE_USD", "25"))
STOP_BUYING_DRAWDOWN_PCT = float(env("STOP_BUYING_DRAWDOWN_PCT", "40"))  # noodrem in code
GOAL_PCT = float(env("GOAL_PCT", "20"))             # doel per periode
GOAL_DAYS = int(env("GOAL_DAYS", "28"))             # lengte van een periode
GOAL_STEP_PCT = float(env("GOAL_STEP_PCT", "10"))   # doel gehaald -> lat zoveel hoger
RISK_TOLERANCE_PCT = float(env("RISK_TOLERANCE_PCT", "25"))  # acceptabele daling vanaf de top (instructie aan AI)
FEE_PCT = float(env("FEE_PCT", "0.25"))
AI_MAX_TOKENS = int(env("AI_MAX_TOKENS", "8000"))
AI_TIMEOUT = int(env("AI_TIMEOUT", "240"))
HISTORY_IN_PROMPT = int(env("HISTORY_IN_PROMPT", "8"))
REVIEW_EVERY_HOURS = float(env("REVIEW_EVERY_HOURS", "24"))
FNG_URL = "https://api.alternative.me/fng/?limit=8"
NTFY_TOPIC = env("NTFY_TOPIC", "")                  # meldingen op je telefoon via de ntfy-app
NOTIFY = env("NOTIFY", "all").lower()               # all = elke run, trades = alleen bij orders/mijlpaal/fout, off
DRY_RUN = env("DRY_RUN", "false").lower() in ("1", "true", "yes", "ja")
KRONOS = env("KRONOS", "on").lower() not in ("0", "off", "false", "no", "nee", "uit")
KRONOS_MODEL = env("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER = env("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DIR = env("KRONOS_DIR", str(ROOT / ".kronos"))      # map met de Kronos-code (workflow haalt die op)
KRONOS_LOOKBACK = int(env("KRONOS_LOOKBACK", "400"))       # aantal uurcandles dat Kronos per munt ziet (max 512)
KRONOS_SAMPLES = int(env("KRONOS_SAMPLES", "8"))           # voorspellingen per munt (voor de kans op stijging)
KRONOS_BUDGET_S = int(env("KRONOS_BUDGET_S", "240"))       # rekentijd; daarna vallen de minst belangrijke munten af


# ------------------------------- HTTP --------------------------------------
def http(method, url, headers=None, body=None, timeout=30, retries=2):
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Accept": "application/json", "User-Agent": "ai-trader/1.0"}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    short = url.split("?")[0]
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} bij {method} {short}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"Netwerkfout bij {method} {short}: {e}") from None
    raise RuntimeError(f"Mislukt: {method} {short}")


def qs(params):
    return urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, safe="/,:")


# ------------------------------ Alpaca -------------------------------------
class Alpaca:
    def __init__(self, key, secret):
        self.h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _get(self, path, base=PAPER_URL, **params):
        url = f"{base}{path}"
        if params:
            url += "?" + qs(params)
        return http("GET", url, self.h)

    def account(self):
        return self._get("/v2/account")

    def positions(self):
        return self._get("/v2/positions")

    def crypto_assets(self):
        return self._get("/v2/assets", status="active", asset_class="crypto")

    def bars(self, symbols, timeframe, start):
        out, token = {}, None
        while True:
            res = self._get("/bars", base=DATA_URL, symbols=",".join(symbols), timeframe=timeframe,
                            start=start, limit=10000, sort="asc", page_token=token)
            for sym, rows in (res.get("bars") or {}).items():
                out.setdefault(sym, []).extend(rows)
            token = res.get("next_page_token")
            if not token:
                return out

    def order(self, symbol, side, notional=None, qty=None):
        body = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "gtc"}
        if notional is not None:
            body["notional"] = format(Decimal(str(notional)), "f")
        else:
            body["qty"] = format(Decimal(str(qty)), "f")
        return http("POST", f"{PAPER_URL}/v2/orders", self.h, body, retries=0)

    def close(self, symbol_or_id):
        return http("DELETE", f"{PAPER_URL}/v2/positions/{urllib.parse.quote(symbol_or_id, safe='')}", self.h, retries=0)


# ---------------------------- Indicatoren ----------------------------------
def pct(a, b):
    return (a / b - 1) * 100 if a is not None and b else None


def sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    d = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    g = sum(max(x, 0) for x in d[:n]) / n
    l = sum(max(-x, 0) for x in d[:n]) / n
    for x in d[n:]:
        g = (g * (n - 1) + max(x, 0)) / n
        l = (l * (n - 1) + max(-x, 0)) / n
    if l == 0:
        return 100.0
    return 100 - 100 / (1 + g / l)


def stdev(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def features(hourly, daily):
    h = [float(b["c"]) for b in hourly]
    d = [float(b["c"]) for b in daily]
    if len(h) < 25:
        return None
    price = h[-1]
    rets = [(h[i] / h[i - 1] - 1) * 100 for i in range(max(1, len(h) - 24), len(h))]
    lo7, hi7 = min(h), max(h)
    usd_vol = sum(float(b.get("v", 0)) * float(b["c"]) for b in hourly[-24:])
    return {
        "price": price,
        "ch_1h": pct(price, h[-2]),
        "ch_24h": pct(price, h[-25]),
        "ch_7d": pct(price, h[0]),
        "ch_30d": pct(price, d[-31]) if len(d) >= 31 else None,
        "rsi_1h": rsi(h),
        "rsi_1d": rsi(d),
        "vs_sma24h": pct(price, sma(h, 24)),
        "vs_sma20d": pct(price, sma(d, 20)),
        "vs_sma50d": pct(price, sma(d, 50)),
        "range7d_pos": (price - lo7) / (hi7 - lo7) * 100 if hi7 > lo7 else 50.0,
        "vol_1h": stdev(rets),
        "usd_vol_24h": usd_vol,
    }


FEATURE_COLS = ["price", "ch_1h", "ch_24h", "ch_7d", "ch_30d", "rsi_1h", "rsi_1d",
                "vs_sma24h", "vs_sma20d", "vs_sma50d", "range7d_pos", "vol_1h", "usd_vol_24h"]


def fmt_num(v, col):
    if v is None:
        return "na"
    if col == "price":
        return f"{v:.6g}" if v < 1 else f"{v:.2f}"
    if col == "usd_vol_24h":
        return f"{v:.0f}"
    return f"{v:.2f}"


# ------------------------------- AI ----------------------------------------
SYSTEM_PROMPT = """You are the autonomous portfolio manager of a crypto trading experiment (paper account, simulated money).
Every 30 minutes you receive the portfolio, market data, the outcomes of your recent decisions, your own strategy plan
and the lessons from your daily self-reviews. You decide the target allocation.

YOUR MANDATE
You have full trust and full discretion over strategy: trend following, buying dips, rotating between coins,
concentrating in your single best idea, or sitting in cash. Pick whatever you judge gives the best result and change
approach whenever the market changes. Trust your own analysis and act decisively when you see an edge.

THE GOAL
- Grow the portfolio by at least the current period target (shown below) before the period ends, and beat simply
  holding BTC. This is an ambitious target on purpose: it requires taking real, deliberate risk.
- Drawdowns of up to about {risk:.0f}% from the peak are acceptable in pursuit of the goal. Losses beyond that are not.
- Reaching the target is a milestone, not the finish line. When you reach it, the bar is raised automatically.
  Do not coast and do not retreat into cash just to protect the number. Keep hunting for the best risk/reward with the
  same focus and intensity. Protect gains only when the market itself gives you a reason to.

HARD LIMITS (enforced by code)
- Spot only: no leverage, no shorting.
- Max {max_coin:.0f}% of equity per coin. Minimum cash: {min_cash:.0f}%.
- Each trade costs about {fee:.2f}% in fees plus spread, and changes smaller than {band:.0f}% of equity are ignored.
  Trade when it is worth it: needless churn eats returns.
- Coins NOT listed in "allocations" are fully sold. Only use symbols from the market data table.

LEARN AS YOU GO
Look at what your recent decisions actually produced. Double down on what works, drop what doesn't, and update your
plan whenever your view changes. Apply the lessons from your self-reviews.
{kronos}
Respond with ONLY one JSON object, no markdown, in exactly this shape:
{{"summary": "<2-3 sentences market view, in Dutch>",
  "plan": "<your current strategy and what you expect over the coming hours/days, max 600 characters, in Dutch>",
  "allocations": [{{"symbol": "BTC/USD", "target_pct": 40, "reason": "<short reason, in Dutch>"}}]}}"""

KRONOS_RULES = """
KRONOS FORECASTS
You also receive price forecasts from Kronos, an open-source time-series model for candlesticks. Treat them as a hint,
never as truth: the model is often wrong and knows nothing about news, sentiment or the wider market. Weigh it by its
measured track record (around 50% means it is no better than a coin flip). Never trade on a Kronos forecast alone. It may
tip the balance when it agrees with price action and your plan; a strong disagreement is a reason to double-check your
view, not to flip it.
"""

REVIEW_PROMPT = """You are the autonomous portfolio manager of a crypto trading experiment (paper account). Once a day you
review your own recent performance to get better. Be honest, specific and practical. Look at which decisions made or
lost money and why, how you did versus simply holding BTC, and whether you are on track for the goal.

You receive your existing lessons. Keep the ones that are still valid, sharpen or merge them, drop the ones that proved
wrong, and add new ones. Maximum 8 lessons, each one concrete and actionable.

Respond with ONLY one JSON object, no markdown:
{"assessment": "<2-3 sentences honest assessment, in Dutch>", "lessons": ["<lesson, in Dutch>", "..."]}"""


def parse_ai_json(text, keys=("allocations", "targets")):
    """Haalt het JSON-object met een van de gevraagde sleutels uit het antwoord (ook als er tekst omheen staat)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    dec = json.JSONDecoder()
    best, i = None, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and any(k in obj for k in keys):
            best = obj
            i = text.find("{", end)
        else:
            i = text.find("{", i + 1)
    return best


def ask_ai(nv_key, system, user, keys=("allocations", "targets"), budget_min=14):
    errors, deadline = [], time.time() + budget_min * 60
    for model in MODELS:
        if time.time() > deadline:
            errors.append(f"{model}: overgeslagen (tijdslimiet)")
            continue
        try:
            res = http("POST", NVIDIA_URL, {"Authorization": f"Bearer {nv_key}"}, {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.4,
                "top_p": 0.95,
                "max_tokens": AI_MAX_TOKENS,
                "stream": False,
            }, timeout=AI_TIMEOUT, retries=1)
            msg = (res.get("choices") or [{}])[0].get("message") or {}
            parsed = (parse_ai_json(msg.get("content") or "", keys)
                      or parse_ai_json(msg.get("reasoning_content") or "", keys))
            if parsed is None:
                raise ValueError("antwoord bevatte geen geldige JSON")
            return model, parsed, errors
        except Exception as e:  # volgende model proberen
            errors.append(f"{model}: {str(e)[:200]}")
    return None, None, errors


def norm_symbol(raw, universe):
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper().replace("-", "/")
    if s in universe:
        return s
    if "/" not in s:
        s2 = s[:-3] + "/USD" if s.endswith("USD") and s[:-3] + "/USD" in universe else s + "/USD"
        if s2 in universe:
            return s2
    return None


def sanitize(parsed, universe):
    """AI-antwoord -> veilige doelverdeling {symbol: pct}."""
    notes, targets, reasons = [], {}, {}
    allocs = parsed.get("allocations", parsed.get("targets", []))
    if isinstance(allocs, dict):
        allocs = [{"symbol": k, "target_pct": v} for k, v in allocs.items()]
    if not isinstance(allocs, list):
        raise ValueError("'allocations' is geen lijst")
    for a in allocs:
        if not isinstance(a, dict):
            continue
        sym = norm_symbol(a.get("symbol"), universe)
        if sym is None:
            notes.append(f"onbekende munt {a.get('symbol')!r} genegeerd")
            continue
        try:
            p = float(a.get("target_pct", 0))
        except (TypeError, ValueError):
            notes.append(f"{sym}: ongeldig percentage genegeerd")
            continue
        if not math.isfinite(p) or p <= 0:
            continue
        targets[sym] = targets.get(sym, 0) + p
        reasons[sym] = str(a.get("reason", ""))[:300]
    for sym, p in list(targets.items()):
        if p > MAX_PER_COIN_PCT:
            notes.append(f"{sym}: {p:.0f}% verlaagd naar max {MAX_PER_COIN_PCT:.0f}%")
            targets[sym] = MAX_PER_COIN_PCT
    total, cap = sum(targets.values()), 100 - MIN_CASH_PCT
    if total > cap:
        f = cap / total
        targets = {k: v * f for k, v in targets.items()}
        notes.append(f"totaal {total:.0f}% verlaagd naar {cap:.0f}% (min. {MIN_CASH_PCT:.0f}% cash)")
    return targets, reasons, notes


# ----------------------------- Handelen ------------------------------------
def floor_to(q, inc):
    inc = Decimal(str(inc)) if inc else Decimal("0.000000001")
    if inc <= 0:
        inc = Decimal("0.000000001")
    return (Decimal(str(q)) / inc).to_integral_value(rounding=ROUND_DOWN) * inc


def plan_trades(equity, holdings, targets):
    band = max(MIN_TRADE_USD, equity * REBALANCE_BAND_PCT / 100)
    sells, buys = [], []
    for sym in sorted(set(holdings) | set(targets)):
        cur = holdings.get(sym, {}).get("value", 0.0)
        tgt_pct = targets.get(sym, 0.0)
        tgt = equity * tgt_pct / 100
        if tgt_pct == 0 and cur >= 1:
            sells.append({"symbol": sym, "kind": "close", "usd": cur})
        elif tgt - cur <= -band:
            sells.append({"symbol": sym, "kind": "sell", "usd": cur - tgt})
        elif tgt - cur >= band:
            buys.append({"symbol": sym, "kind": "buy", "usd": tgt - cur})
    buys.sort(key=lambda t: -t["usd"])
    return sells, buys


def execute(alpaca, sells, buys, holdings, prices, assets, buying_power):
    results = []
    for t in sells:
        sym, pos = t["symbol"], holdings.get(t["symbol"], {})
        try:
            if t["kind"] == "close" or t["usd"] >= 0.98 * pos.get("value", 0):
                if DRY_RUN:
                    res = "proefrun"
                else:
                    alpaca.close(pos.get("asset_id") or sym.replace("/", ""))
                    res = "verkocht (alles)"
                results.append({**t, "kind": "close", "result": res})
                continue
            a = assets.get(sym, {})
            price = pos.get("price") or prices.get(sym)
            if not price:
                results.append({**t, "result": "overgeslagen (geen koers)"})
                continue
            qty = floor_to(min(t["usd"] / price, pos.get("qty_available", pos.get("qty", 0))), a.get("min_trade_increment"))
            if qty <= 0 or qty < Decimal(str(a.get("min_order_size") or 0)):
                results.append({**t, "result": "overgeslagen (te klein)"})
                continue
            if DRY_RUN:
                res = "proefrun"
            else:
                alpaca.order(sym, "sell", qty=qty)
                res = f"verkocht {qty}"
            results.append({**t, "result": res})
        except Exception as e:
            results.append({**t, "result": f"FOUT: {str(e)[:150]}"})

    if sells and buys and not DRY_RUN:
        time.sleep(4)
        try:
            acct = alpaca.account()
            buying_power = float(acct.get("non_marginable_buying_power") or acct.get("cash") or 0)
        except Exception:
            pass
    elif DRY_RUN:
        buying_power += sum(t["usd"] for t in sells)

    available = buying_power * (1 - FEE_PCT / 100) * 0.99
    for t in buys:
        sym, a = t["symbol"], assets.get(t["symbol"], {})
        notional = Decimal(str(min(t["usd"], available))).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if notional < Decimal(str(MIN_TRADE_USD)):
            results.append({**t, "result": "overgeslagen (te weinig cash)"})
            continue
        price = prices.get(sym)
        if price and float(notional) / price < float(a.get("min_order_size") or 0):
            results.append({**t, "result": "overgeslagen (onder minimum)"})
            continue
        try:
            if DRY_RUN:
                res = "proefrun"
            else:
                alpaca.order(sym, "buy", notional=notional)
                res = f"gekocht ${notional}"
            available -= float(notional)
            results.append({**t, "usd": float(notional), "result": res})
        except Exception as e:
            results.append({**t, "result": f"FOUT: {str(e)[:150]}"})
    return results


# ------------------------------ Opslag -------------------------------------
def load_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    DATA_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path, header, rows):
    DATA_DIR.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def read_csv_tail(path, n):
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))[-n:]
    except Exception:
        return []


def local(dt):
    return dt.astimezone(TZ).strftime("%d-%m-%Y %H:%M")


def write_status(md):
    README.write_text(md, encoding="utf-8")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md)


def render_status(now, equity, cash, start, ret, hold_ret, holdings, targets, reasons, summary,
                  model, results, notes, errors, state=None, period_end=None, fng=None,
                  kronos=None, kronos_rec=(0, 0), kronos_status=""):
    sign = lambda v: "n.v.t." if v is None else f"{v:+.2f}%"
    kronos = kronos or {}
    kronos_row = []
    if KRONOS:
        hits, total = kronos_rec
        rec = f"richting op 4u goed: {hits / total * 100:.0f}% (n={total}, 50% = gokken)" if total else "trefzekerheid nog niet gemeten"
        kronos_row = [f"| Kronos-voorspelling | {kronos_status}; {rec} |"]
    state = state or {}
    g = state.get("goal")
    goal_rows = []
    if g:
        pr = pct(equity, g["start_equity"])
        reached = ", ".join(f"+{m['target_pct']:.0f}%" for m in g["milestones"]) or "nog geen"
        goal_rows = [
            f"| Doel periode {g['period']} (t/m {local(period_end)[:10] if period_end else '?'}) | **+{g['target_pct']:.0f}%** |",
            f"| Rendement deze periode | **{sign(pr)}** |",
            f"| Gehaalde mijlpalen | {reached} |",
        ]
    peak = state.get("peak_equity")
    lines = [
        "# AI Crypto Trader (demogeld)",
        "",
        f"_Laatst bijgewerkt: {local(now)}{' - PROEFRUN, geen orders geplaatst' if DRY_RUN else ''}_",
        "",
        "| | |",
        "|---|---|",
        f"| Waarde portefeuille | **${equity:,.2f}** |",
        f"| Cash | ${cash:,.2f} ({cash / equity * 100 if equity else 0:.0f}%) |",
        f"| Rendement AI sinds {local(datetime.fromisoformat(start['time']))} | **{sign(ret)}** |",
        f"| Ter vergelijking: gewoon BTC vasthouden | {sign(hold_ret)} |",
        f"| Verschil | {sign(None if ret is None or hold_ret is None else ret - hold_ret)} |",
        *goal_rows,
        *([f"| Daling vanaf de top (${peak:,.0f}) | {sign(pct(equity, peak))} |"] if peak else []),
        f"| Model | {model or 'geen (AI niet bereikbaar)'} |",
        *([f"| Fear & Greed Index ([alternative.me](https://alternative.me/crypto/fear-and-greed-index/)) | "
           f"{fng['value']} ({fng['label']}) |"] if fng else []),
        *kronos_row,
        "",
    ]
    if summary:
        lines += ["## Marktvisie van de AI", "", summary, ""]
    plan = state.get("plan")
    if plan:
        lines += [f"## Strategie van de AI (sinds {plan['time_local']})", "", plan["text"], ""]
    if state.get("lessons"):
        lines += ["## Lessen uit de dagelijkse evaluatie", ""]
        if state.get("review_assessment"):
            lines += [f"_{state['review_assessment']}_", ""]
        lines += [f"- {l}" for l in state["lessons"]] + [""]
    lines += ["## Posities", ""]
    if holdings:
        lines += ["| Munt | Waarde | % | Winst/verlies | Doel | Kronos 12u | Reden AI |", "|---|---|---|---|---|---|---|"]
        for sym, h in sorted(holdings.items(), key=lambda x: -x[1]["value"]):
            k = kronos.get(sym)
            kt = f"{k['k12h']:+.1f}% ({k['up12h']:.0f}% omhoog)" if k else "-"
            lines.append(f"| {sym} | ${h['value']:,.2f} | {h['value'] / equity * 100:.1f}% | {h.get('pl_pct', 0):+.1f}% "
                         f"| {targets.get(sym, 0):.1f}% | {kt} | {reasons.get(sym, '').replace('|', '/')} |")
    else:
        lines.append("Geen posities (alles in cash).")
    new_targets = {s: p for s, p in targets.items() if s not in holdings and p >= 0.05}
    if new_targets:
        lines += ["", "**Nieuwe doelen:** " + ", ".join(f"{s} {p:.1f}% ({reasons.get(s, '')})" for s, p in new_targets.items())]
    lines += ["", "## Orders deze run", ""]
    if results:
        lines += ["| Munt | Actie | Bedrag | Resultaat |", "|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r['symbol']} | {r['kind']} | ${r['usd']:,.2f} | {r['result']} |")
    else:
        lines.append("Geen orders (verdeling blijft gelijk).")
    if notes or errors:
        lines += ["", "## Meldingen", ""] + [f"- {n}" for n in notes + errors]
    tail = read_csv_tail(EQUITY_CSV, 12)
    if tail:
        lines += ["", "## Laatste runs", "", "| Tijd | Waarde | Rendement | BTC vasthouden | Orders |", "|---|---|---|---|---|"]
        for r in reversed(tail):
            lines.append(f"| {r['time_local']} | ${float(r['equity']):,.0f} | {r['return_pct']}% | {r['btc_hold_pct']}% | {r['trades']} |")
    lines += ["", "---", "Volledig logboek: [`data/equity.csv`](data/equity.csv) en [`data/decisions.csv`](data/decisions.csv). "
              "Installatie en instellingen: [`INSTALLATIE.md`](INSTALLATIE.md).", ""]
    return "\n".join(lines)


# ----------------------------- Meldingen -----------------------------------
def post_text(url, headers, body):
    req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def notify(equity, ret, hold_ret, goal, period_ret, results, notes, errors, summary, model):
    """Stuurt een pushmelding via ntfy.sh. Fouten hierin stoppen de bot nooit."""
    if not NTFY_TOPIC or NOTIFY == "off":
        return
    milestone = any("Mijlpaal" in n for n in notes)
    if NOTIFY == "trades" and not (results or milestone or errors or not model):
        return
    try:
        acts = []
        for r in results:
            verb = {"buy": "koop", "sell": "verkoop deel", "close": "verkoop alles"}.get(r["kind"], r["kind"])
            acts.append(f"{verb} {r['symbol'].split('/')[0]} ${r['usd']:,.0f}" + ("" if not r["result"].startswith("FOUT") else " (FOUT)"))
        lines = [
            f"Rendement {ret:+.2f}% | BTC vasthouden {hold_ret:+.2f}%",
            f"Doel periode {goal['period']}: +{goal['target_pct']:.0f}% (nu {period_ret:+.2f}%)",
            "Orders: " + ("; ".join(acts) if acts else "geen"),
        ]
        lines += [n for n in notes if "Mijlpaal" in n or "Periode" in n or "Noodrem" in n or "kopen gestopt" in n]
        if not model:
            lines.append("Geen AI-antwoord: niets gedaan.")
        if errors:
            lines.append(f"Meldingen: {len(errors)} (zie statuspagina)")
        if summary:
            lines.append(summary[:200])
        tags = ["trophy"] if milestone else ["warning"] if (errors or not model) else \
            ["chart_with_upwards_trend"] if ret >= 0 else ["chart_with_downwards_trend"]
        headers = {"Title": f"AI Trader{' (proefrun)' if DRY_RUN else ''}: ${equity:,.0f} ({ret:+.1f}%)",
                   "Tags": ",".join(tags), "Priority": "4" if milestone else "3"}
        repo = os.getenv("GITHUB_REPOSITORY")
        if repo:
            headers["Click"] = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{repo}"
        post_text(f"https://ntfy.sh/{NTFY_TOPIC}", headers, "\n".join(lines))
    except Exception as e:
        print(f"Melding versturen mislukt: {e}")


# ------------------------- Context & leren ---------------------------------
def fear_greed():
    """Crypto Fear & Greed Index (bron: alternative.me). Geeft None bij een fout."""
    try:
        d = http("GET", FNG_URL, timeout=15, retries=1).get("data") or []
        if not d:
            return None
        cur = d[0]
        prev = d[1]["value"] if len(d) > 1 else "na"
        week = d[7]["value"] if len(d) > 7 else d[-1]["value"]
        return {"value": cur["value"], "label": cur.get("value_classification", ""), "prev": prev, "week": week}
    except Exception:
        return None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def breadth(feats):
    ch24 = [f["ch_24h"] for f in feats.values() if f["ch_24h"] is not None]
    up = sum(1 for x in ch24 if x > 0)
    return (f"{up} of {len(ch24)} coins up over 24h; median 24h change {median(ch24):+.2f}%; "
            f"median 7d change {median([f['ch_7d'] for f in feats.values()]):+.2f}%") if ch24 else "na"


def run_kronos(state, hourly, now, feats, holdings):
    """Kronos-voorspelling per munt + trefzekerheid. Gaat er iets mis, dan gaat de bot gewoon door zonder.
    Geeft (voorspellingen of None, trefzekerheid (goed, totaal), statustekst voor de statuspagina)."""
    import kronos_signal as ks
    record = ks.track_record(state, hourly, now)
    order = sorted(feats, key=lambda s: (s not in holdings, -feats[s]["usd_vol_24h"]))
    try:
        fc, info = ks.forecast(hourly, now, order, KRONOS_DIR, KRONOS_MODEL, KRONOS_TOKENIZER,
                               KRONOS_LOOKBACK, KRONOS_SAMPLES, KRONOS_BUDGET_S)
    except Exception as e:
        return None, record, f"niet beschikbaar: {type(e).__name__}: {str(e)[:150]}"
    if not fc:
        return None, record, "geen munten met genoeg koersdata"
    ks.log_forecast(state, info["bar_t"], fc)
    status = f"{len(fc)} munten in {info['secs']:.0f}s"
    if info["skipped"]:
        status += f" ({len(info['skipped'])} overgeslagen: te weinig handel of tijdslimiet)"
    return fc, record, status


def kronos_record_line(record):
    import kronos_signal as ks
    hits, total = record
    if not total:
        return f"Kronos track record: not measured yet (first result about {ks.EVAL_HORIZON + 1}h after enabling)."
    return (f"Kronos track record: over the last {ks.EVAL_WINDOW_H}h its {ks.EVAL_HORIZON}h direction was right in "
            f"{hits / total * 100:.1f}% of {total} coin forecasts (50% = coin flip).")


def new_period(n, now, equity, btc):
    return {"period": n, "period_start": now.isoformat(), "start_equity": equity, "btc_start": btc,
            "target_pct": GOAL_PCT, "milestones": []}


def update_goal(state, now, equity, btc):
    """Houdt het doel per periode bij. Doel gehaald -> lat omhoog. Periode voorbij -> nieuwe periode."""
    events = []
    g = state.get("goal") or new_period(1, now, equity, btc)
    end = datetime.fromisoformat(g["period_start"]) + timedelta(days=GOAL_DAYS)
    if now >= end:
        r = pct(equity, g["start_equity"])
        state.setdefault("goal_history", []).append({
            "period": g["period"], "start": g["period_start"], "end": now.isoformat(),
            "return_pct": round(r, 2), "btc_pct": round(pct(btc, g["btc_start"]), 2),
            "milestones": [m["target_pct"] for m in g["milestones"]]})
        events.append(f"Periode {g['period']} afgerond met {r:+.1f}%. Nieuwe periode gestart.")
        g = new_period(g["period"] + 1, now, equity, btc)
        end = now + timedelta(days=GOAL_DAYS)
    r = pct(equity, g["start_equity"])
    while r >= g["target_pct"]:
        g["milestones"].append({"time": now.isoformat(), "target_pct": g["target_pct"]})
        events.append(f"Mijlpaal +{g['target_pct']:.0f}% gehaald! De lat gaat omhoog naar +{g['target_pct'] + GOAL_STEP_PCT:.0f}%.")
        g["target_pct"] += GOAL_STEP_PCT
    state["goal"] = g
    return g, r, end, events


def feedback_lines(history, equity, prices):
    """Per eerdere beslissing: wat leverde het op sinds dat moment."""
    lines = []
    for h in history:
        hp = h.get("prices", {})
        parts = []
        for s, p in sorted(h.get("targets", {}).items(), key=lambda x: -x[1])[:8]:
            ch = pct(prices.get(s), hp.get(s))
            parts.append(f"{s} {p:.0f}%" + (f" ({ch:+.1f}% since)" if ch is not None else ""))
        since = pct(equity, h.get("equity"))
        btc = pct(prices.get("BTC/USD"), hp.get("BTC/USD"))
        perf = (f" | portfolio since {since:+.1f}% vs BTC {btc:+.1f}%" if since is not None and btc is not None else "")
        lines.append(f"- {h['time_local']}: {', '.join(parts) or 'all cash'}{perf} | view: {h.get('summary', '')[:160]}")
    return lines


def maybe_review(state, now, nv_key, equity, prices, goal_line, extra=""):
    """Eens per REVIEW_EVERY_HOURS: de AI evalueert zichzelf en werkt zijn lessen bij."""
    last = state.get("last_review")
    if not last:
        state["last_review"] = now.isoformat()
        return None
    if now - datetime.fromisoformat(last) < timedelta(hours=REVIEW_EVERY_HOURS):
        return None
    since = now - timedelta(hours=REVIEW_EVERY_HOURS + 2)
    hist = [h for h in state.get("history", []) if datetime.fromisoformat(h["time"]) >= since]
    if len(hist) < 6:
        return None
    first = hist[0]
    lessons = state.get("lessons", [])
    user = "\n".join([
        f"Time (UTC): {now.strftime('%Y-%m-%d %H:%M')}",
        goal_line,
        f"Portfolio over this review window: {pct(equity, first.get('equity')):+.2f}% | "
        f"BTC buy-and-hold same window: {pct(prices.get('BTC/USD'), first.get('prices', {}).get('BTC/USD')):+.2f}%",
        *([extra] if extra else []),
        "",
        "YOUR DECISIONS IN THIS WINDOW AND WHAT THEY PRODUCED (oldest first):",
        *feedback_lines(hist, equity, prices),
        "",
        "YOUR CURRENT LESSONS:",
        *([f"- {l}" for l in lessons] or ["- none yet"]),
    ])
    model, parsed, errs = ask_ai(nv_key, REVIEW_PROMPT, user, keys=("lessons",), budget_min=8)
    if not parsed or not isinstance(parsed.get("lessons"), list):
        return f"Dagelijkse evaluatie mislukt: {'; '.join(errs)[:200]}"
    state["lessons"] = [str(l)[:250] for l in parsed["lessons"] if str(l).strip()][:8]
    state["review_assessment"] = str(parsed.get("assessment", ""))[:600]
    state["last_review"] = now.isoformat()
    return f"Dagelijkse evaluatie gedaan door {model}."


# ------------------------------- Main --------------------------------------
def get_holdings(alpaca, nosl):
    holdings = {}
    for p in alpaca.positions():
        if p.get("asset_class") != "crypto":
            continue
        raw = p["symbol"]
        sym = nosl.get(raw.replace("/", ""), raw if "/" in raw else raw[:-3] + "/USD")
        holdings[sym] = {
            "value": float(p.get("market_value") or 0),
            "qty": float(p.get("qty") or 0),
            "qty_available": float(p.get("qty_available") or p.get("qty") or 0),
            "price": float(p.get("current_price") or 0),
            "pl_pct": float(p.get("unrealized_plpc") or 0) * 100,
            "asset_id": p.get("asset_id"),
        }
    return holdings


def main():
    key, secret, nv_key = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), os.getenv("NVIDIA_API_KEY")
    missing = [n for n, v in [("ALPACA_API_KEY", key), ("ALPACA_SECRET_KEY", secret), ("NVIDIA_API_KEY", nv_key)] if not v]
    if missing:
        print(f"Ontbrekende sleutels (GitHub Secrets): {', '.join(missing)}")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    alpaca = Alpaca(key, secret)
    state = load_state()
    notes, errors = [], []

    # 1. Portefeuille
    acct = alpaca.account()
    if acct.get("trading_blocked") or acct.get("account_blocked"):
        print("Account is geblokkeerd voor handel.")
        sys.exit(1)
    equity = float(acct["equity"])
    cash = float(acct["cash"])
    buying_power = float(acct.get("non_marginable_buying_power") or cash)

    # 2. Munten
    assets = {}
    for a in alpaca.crypto_assets():
        sym = a.get("symbol", "")
        if a.get("tradable") and sym.endswith("/USD") and sym.split("/")[0] not in STABLECOINS:
            assets[sym] = a
    if SYMBOLS.upper() != "ALL":
        wanted = {s.strip().upper() for s in SYMBOLS.split(",")}
        assets = {s: a for s, a in assets.items() if s in wanted}
    nosl = {s.replace("/", ""): s for s in assets}
    holdings = get_holdings(alpaca, nosl)

    # 3. Marktdata
    symbols = sorted(set(assets) | {s for s in holdings if s in assets})
    week_ago = now - timedelta(days=7, hours=1)
    hours = max(7 * 24 + 1, KRONOS_LOOKBACK + 2) if KRONOS else 7 * 24 + 1   # Kronos wil een langere geschiedenis
    hourly = alpaca.bars(symbols, "1Hour", (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    daily = alpaca.bars(symbols, "1Day", (now - timedelta(days=61)).strftime("%Y-%m-%d"))
    feats = {}
    for s in symbols:
        last_week = [b for b in hourly.get(s, []) if datetime.fromisoformat(b["t"].replace("Z", "+00:00")) >= week_ago]
        f = features(last_week, daily.get(s, []))
        if f:
            feats[s] = f
    prices = {s: f["price"] for s, f in feats.items()}
    universe = set(feats)
    if "BTC/USD" not in prices:
        raise RuntimeError("Geen BTC-koers ontvangen")

    # Startpunt (voor rendement en vergelijking met BTC vasthouden)
    start = state.get("start") or {"time": now.isoformat(), "equity": equity, "btc_price": prices["BTC/USD"]}
    ret = pct(equity, start["equity"])
    hold_ret = pct(prices["BTC/USD"], start["btc_price"])

    # Doel, top en daling vanaf de top
    goal, period_ret, period_end, goal_events = update_goal(state, now, equity, prices["BTC/USD"])
    notes += goal_events
    period_btc = pct(prices["BTC/USD"], goal["btc_start"])
    peak = max(state.get("peak_equity", equity), equity)
    state["peak_equity"] = peak
    dd = pct(equity, peak)
    day = (now - datetime.fromisoformat(goal["period_start"])).days + 1
    reached = [m["target_pct"] for m in goal["milestones"]]
    goal_line = (f"GOAL period {goal['period']} (day {day} of {GOAL_DAYS}, ends {period_end.strftime('%Y-%m-%d')}): "
                 f"reach +{goal['target_pct']:.0f}% this period. Period return so far {period_ret:+.2f}% "
                 f"| BTC buy-and-hold this period {period_btc:+.2f}%.")
    if reached:
        goal_line += (f" Milestones already reached this period: {', '.join(f'+{m:.0f}%' for m in reached)}. "
                      f"The bar has been raised to +{goal['target_pct']:.0f}%. Stay sharp; do not coast.")

    # Kronos-voorspelling (extra informatiebron voor de AI, geen beslisser)
    kronos, kronos_rec, kronos_status = None, (0, 0), ""
    if KRONOS:
        kronos, kronos_rec, kronos_status = run_kronos(state, hourly, now, feats, holdings)
        if kronos is None:
            notes.append(f"Kronos {kronos_status}")

    # Dagelijkse zelfevaluatie (bijwerken van lessen)
    if not DRY_RUN:
        msg = maybe_review(state, now, nv_key, equity, prices, goal_line,
                           kronos_record_line(kronos_rec) if KRONOS else "")
        if msg:
            notes.append(msg)

    # 4. Vraag de AI
    table = ["symbol," + ",".join(FEATURE_COLS)]
    for s in sorted(feats, key=lambda x: -feats[x]["usd_vol_24h"]):
        table.append(s + "," + ",".join(fmt_num(feats[s][c], c) for c in FEATURE_COLS))
    pos_lines = [f"- {s}: {h['value'] / equity * 100:.1f}% (${h['value']:,.2f}), P/L {h['pl_pct']:+.1f}%"
                 for s, h in sorted(holdings.items(), key=lambda x: -x[1]["value"])] or ["- none (all cash)"]
    hist_lines = feedback_lines(state.get("history", [])[-HISTORY_IN_PROMPT:], equity, prices) or ["- none yet (first run)"]
    fng = fear_greed()
    fng_line = (f"Crypto Fear & Greed Index (0 = extreme fear, 100 = extreme greed): {fng['value']} ({fng['label']}), "
                f"yesterday {fng['prev']}, 7 days ago {fng['week']}") if fng else "Crypto Fear & Greed Index: unavailable"
    kronos_lines = []
    if KRONOS:
        kronos_lines = ["", "KRONOS FORECAST (hint, not truth)", kronos_record_line(kronos_rec)]
        if kronos:
            kronos_lines += [
                f"Kronos only sees each coin's own last {KRONOS_LOOKBACK} hourly candles: no news, no other coins.",
                f"k4h/k12h = mean forecast % change vs the last closed hourly candle; up4h/up12h = % of "
                f"{KRONOS_SAMPLES} sampled forecast paths that end higher.",
                "symbol,k4h,k12h,up4h,up12h",
                *[f"{s},{v['k4h']:+.2f},{v['k12h']:+.2f},{v['up4h']:.0f},{v['up12h']:.0f}"
                  for s, v in sorted(kronos.items(), key=lambda x: -feats[x[0]]["usd_vol_24h"])],
            ]
        else:
            kronos_lines.append("No forecast available this run.")
    plan = state.get("plan")
    user_prompt = "\n".join([
        f"Time (UTC): {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        goal_line,
        "",
        "PORTFOLIO",
        f"equity ${equity:,.2f} | cash ${cash:,.2f} ({cash / equity * 100:.1f}%) | peak ${peak:,.2f} | "
        f"drawdown from peak {dd:+.2f}%",
        f"return since the very start: {ret:+.2f}% | BTC buy-and-hold same period: {hold_ret:+.2f}%",
        "positions (pct of equity):",
        *pos_lines,
        "",
        f"YOUR STRATEGY PLAN (written {plan['time_local']}):\n{plan['text']}" if plan else "YOUR STRATEGY PLAN: none yet",
        "",
        "LESSONS FROM YOUR DAILY SELF-REVIEWS:",
        *([f"- {l}" for l in state.get("lessons", [])] or ["- none yet (first review after ~24h)"]),
        "",
        "YOUR RECENT DECISIONS AND WHAT THEY PRODUCED (oldest first):",
        *hist_lines,
        "",
        "MARKET CONTEXT",
        fng_line,
        f"Market breadth: {breadth(feats)}",
        "",
        "MARKET DATA (ch_ = % change, rsi = RSI14, vs_sma = % above/below moving average, "
        "range7d_pos = 0 at 7d low / 100 at 7d high, vol_1h = stdev of hourly returns %, usd_vol_24h = 24h volume in USD):",
        *table,
        *kronos_lines,
    ])
    system_prompt = SYSTEM_PROMPT.format(max_coin=MAX_PER_COIN_PCT, min_cash=MIN_CASH_PCT, fee=FEE_PCT,
                                         band=REBALANCE_BAND_PCT, risk=RISK_TOLERANCE_PCT,
                                         kronos=KRONOS_RULES if KRONOS else "")
    model, parsed, ai_errors = ask_ai(nv_key, system_prompt, user_prompt)
    errors += [f"AI-fout: {e}" for e in ai_errors]

    targets, reasons, summary, new_plan = None, {}, "", ""
    if parsed is not None:
        try:
            targets, reasons, notes2 = sanitize(parsed, universe | set(holdings))
            notes += notes2
            summary = str(parsed.get("summary", ""))[:800]
            new_plan = str(parsed.get("plan", "")).strip()[:700]
        except Exception as e:
            errors.append(f"AI-antwoord onbruikbaar: {e}")
            targets = None

    results = []
    if targets is None:
        notes.append("Geen bruikbaar AI-antwoord: niets gedaan (veilige standaard).")
        targets = {s: h["value"] / equity * 100 for s, h in holdings.items()}
    else:
        # Veiligheidsregel: bij groot verlies alleen nog verkopen
        if ret is not None and ret <= -STOP_BUYING_DRAWDOWN_PCT:
            notes.append(f"Verlies {ret:.1f}% >= {STOP_BUYING_DRAWDOWN_PCT:.0f}%: kopen gestopt, alleen verkopen toegestaan.")
            targets = {s: min(p, holdings.get(s, {}).get("value", 0) / equity * 100) for s, p in targets.items()}
        # 5. Orders
        sells, buys = plan_trades(equity, holdings, targets)
        results = execute(alpaca, sells, buys, holdings, prices, assets, buying_power)

    # Stand na de orders ophalen voor logboek en statuspagina
    holdings_before, equity_before = holdings, equity
    if results and not DRY_RUN:
        time.sleep(4)
        try:
            acct = alpaca.account()
            equity, cash = float(acct["equity"]), float(acct["cash"])
            holdings = get_holdings(alpaca, nosl)
            ret = pct(equity, start["equity"])
        except Exception as e:
            errors.append(f"Stand na orders niet opgehaald: {str(e)[:150]}")

    # 6. Opslaan
    if not DRY_RUN:
        state["start"] = start
        if model:
            tg = {s: round(p, 1) for s, p in targets.items() if p >= 0.05}
            state.setdefault("history", []).append({
                "time": now.isoformat(), "time_local": local(now), "model": model, "targets": tg,
                "summary": summary[:300], "equity": round(equity_before, 2),
                "prices": {s: prices[s] for s in set(tg) | {"BTC/USD"} if s in prices},
            })
            state["history"] = state["history"][-100:]
            if new_plan:
                state["plan"] = {"time": now.isoformat(), "time_local": local(now), "text": new_plan}
        invested = sum(h["value"] for h in holdings.values())
        append_csv(EQUITY_CSV,
                   ["time_utc", "time_local", "equity", "cash", "invested_pct", "return_pct", "btc_price",
                    "btc_hold_pct", "model", "trades", "note"],
                   [[now.isoformat(), local(now), f"{equity:.2f}", f"{cash:.2f}", f"{invested / equity * 100:.1f}",
                     f"{ret:.2f}", f"{prices['BTC/USD']:.2f}", f"{hold_ret:.2f}", model or "", len(results),
                     "; ".join(notes + errors)[:500]]])
        rows = []
        for s in sorted(set(targets) | set(holdings_before) | {r["symbol"] for r in results}):
            r = next((x for x in results if x["symbol"] == s), None)
            rows.append([now.isoformat(), local(now), model or "", s,
                         f"{holdings_before.get(s, {}).get('value', 0) / equity_before * 100:.1f}", f"{targets.get(s, 0):.1f}",
                         r["kind"] if r else "houden", f"{r['usd']:.2f}" if r else "", r["result"] if r else "",
                         reasons.get(s, "")])
        append_csv(DECISIONS_CSV, ["time_utc", "time_local", "model", "symbol", "current_pct", "target_pct",
                                   "action", "usd", "result", "reason"], rows)
        save_state(state)

    shown_state = dict(state)
    if new_plan:
        shown_state["plan"] = {"time_local": local(now), "text": new_plan}
    write_status(render_status(now, equity, cash, start, ret, hold_ret, holdings, targets, reasons, summary,
                               model, results, notes, errors, shown_state, period_end, fng,
                               kronos, kronos_rec, kronos_status))
    notify(equity, ret, hold_ret, goal, pct(equity, goal["start_equity"]), results, notes, errors, summary, model)
    print(f"Klaar. Model: {model}. Waarde ${equity:,.2f} ({ret:+.2f}%), BTC vasthouden {hold_ret:+.2f}%. "
          f"Orders: {len(results)}. Meldingen: {notes + errors}")


if __name__ == "__main__":
    main()
