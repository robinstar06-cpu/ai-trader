"""
Kronos-voorspelling als extra informatiebron voor de AI (optioneel).

Kronos (github.com/shiyu-coder/Kronos, MIT-licentie) is een open-source voorspellingsmodel voor candles.
Per munt krijgt het de laatste uurcandles en voorspelt het de komende uren. Per munt trekken we meerdere
voorspellingen; daaruit volgen de gemiddelde verwachte koersverandering en de kans op stijging (het aandeel
voorspellingen dat hoger eindigt).

Daarnaast houden we bij hoe vaak Kronos de richting goed had, zodat de AI weet hoeveel waarde hij eraan
kan hechten.

Heeft PyTorch en de Kronos-code nodig (de workflow regelt dat). Dit bestand zelf laadt die pas bij gebruik,
dus de bot werkt ook zonder.
"""
import bisect
import os
import sys
import time
from datetime import datetime, timedelta

HORIZONS = (4, 12)      # voorspelling in uren vooruit
EVAL_HORIZON = 4        # op deze horizon meten we de trefzekerheid
EVAL_WINDOW_H = 48      # trefzekerheid over de laatste zoveel uur
LOG_SIZE = 60           # zoveel eerdere voorspellingen bewaren in state.json
BATCH = 64              # reeksen per rekenstap (houdt het geheugengebruik laag)
MIN_REAL_PCT = 80       # munt overslaan als minder dan zoveel % van de uren echte handel had

_predictor = None


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _load(kronos_dir, model_name, tokenizer_name, max_context):
    global _predictor
    if _predictor is None:
        if kronos_dir not in sys.path:
            sys.path.insert(0, kronos_dir)
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer
        torch.set_num_threads(os.cpu_count() or 2)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name).eval()
        model = Kronos.from_pretrained(model_name).eval()
        _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=max_context)
    return _predictor


def last_closed_hour(now):
    """Starttijd van het laatste volledig afgesloten uur (de lopende uurcandle telt nog niet mee)."""
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def _grid(bars, end, n):
    """Alpaca-uurbars -> aaneengesloten reeks van n uren tot en met 'end'.
    Uren zonder handel krijgen de vorige slotkoers en volume 0. Geeft (rijen, aantal echte uren)."""
    by_t = {_ts(b["t"]): b for b in bars}
    times = [end - timedelta(hours=n - 1 - i) for i in range(n)]
    before = [t for t in by_t if t < times[0]]
    prev = float(by_t[max(before)]["c"]) if before else None
    rows, real = [], 0
    for t in times:
        b = by_t.get(t)
        if b:
            o, h, l, c, v = (float(b[k]) for k in ("o", "h", "l", "c", "v"))
            rows.append([o, h, l, c, v, v * float(b.get("vw") or c)])
            real += 1
            prev = c
        else:
            rows.append(None if prev is None else [prev, prev, prev, prev, 0.0, 0.0])
    first = next((r for r in rows if r is not None), None)
    if first is None:
        return None, 0
    rows = [r if r is not None else [first[0]] * 4 + [0.0, 0.0] for r in rows]
    return rows, real


def forecast(hourly, now, symbols, kronos_dir, model_name, tokenizer_name, lookback, samples, budget_s):
    """Voorspelt per munt de koersverandering over HORIZONS uur.
    'symbols' in volgorde van belangrijkheid: bij tijdgebrek vallen de laatste munten af.
    Geeft ({munt: {"k4h", "k12h", "up4h", "up12h"}}, info)."""
    import pandas as pd

    predictor = _load(kronos_dir, model_name, tokenizer_name, max_context=512)
    started = time.time()
    end = last_closed_hour(now)
    pred_len = max(HORIZONS)
    naive_end = end.replace(tzinfo=None)
    x_ts = pd.Series(pd.date_range(end=naive_end, periods=lookback, freq="h"))
    y_ts = pd.Series(pd.date_range(start=naive_end + timedelta(hours=1), periods=pred_len, freq="h"))

    series, thin, late = [], [], []
    for sym in symbols:
        rows, real = _grid(hourly.get(sym, []), end, lookback)
        if rows is None or real * 100 < MIN_REAL_PCT * lookback:
            thin.append(sym)
            continue
        series.append((sym, pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "amount"])))

    out, per_chunk = {}, max(1, BATCH // samples)
    for i in range(0, len(series), per_chunk):
        if time.time() - started > budget_s:
            late = [s for s, _ in series[i:]]
            break
        chunk = series[i:i + per_chunk]
        dfs = [df for _, df in chunk for _ in range(samples)]
        preds = predictor.predict_batch(dfs, [x_ts] * len(dfs), [y_ts] * len(dfs), pred_len=pred_len,
                                        T=1.0, top_p=0.9, sample_count=1, verbose=False)
        for j, (sym, df) in enumerate(chunk):
            base = float(df["close"].iloc[-1])
            paths = preds[j * samples:(j + 1) * samples]
            res = {}
            for h in HORIZONS:
                ch = [(float(p["close"].iloc[h - 1]) / base - 1) * 100 for p in paths]
                res[f"k{h}h"] = round(sum(ch) / len(ch), 3)
                res[f"up{h}h"] = round(sum(1 for c in ch if c > 0) * 100 / len(ch))
            out[sym] = res
    return out, {"bar_t": end.isoformat(), "secs": round(time.time() - started), "thin": thin, "late": late}


def log_forecast(state, bar_t, fc):
    """Bewaart de 4-uursvoorspelling (één keer per uurcandle) om later de trefzekerheid te meten."""
    log = state.setdefault("kronos_log", [])
    if log and log[-1]["t"] == bar_t:
        return
    log.append({"t": bar_t, "f": {s: round(v[f"k{EVAL_HORIZON}h"], 2) for s, v in fc.items()}})
    state["kronos_log"] = log[-LOG_SIZE:]


def track_record(state, hourly, now):
    """Hoe vaak had Kronos de richting goed op EVAL_HORIZON uur, over de laatste EVAL_WINDOW_H uur.
    Geeft (goed, totaal)."""
    closes = {}

    def close_at(sym, t):
        if sym not in closes:
            pts = sorted((_ts(b["t"]), float(b["c"])) for b in hourly.get(sym, []))
            closes[sym] = ([p[0] for p in pts], [p[1] for p in pts])
        ts, cs = closes[sym]
        i = bisect.bisect_right(ts, t) - 1
        return cs[i] if i >= 0 else None

    hits = total = 0
    oldest = now - timedelta(hours=EVAL_WINDOW_H + EVAL_HORIZON + 1)
    for snap in state.get("kronos_log", []):
        t0 = datetime.fromisoformat(snap["t"])
        t1 = t0 + timedelta(hours=EVAL_HORIZON)
        if t0 < oldest or t1 + timedelta(hours=1) > now:
            continue
        for sym, f in snap["f"].items():
            c0, c1 = close_at(sym, t0), close_at(sym, t1)
            if not f or c0 is None or c1 is None or c1 == c0:
                continue
            total += 1
            hits += (c1 > c0) == (f > 0)
    return hits, total
