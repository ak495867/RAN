#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAN-Research: leak-free walk-forward evaluation of a temporal-CNN + graph-conv
cross-sectional stock-ranking model (RAN), with baselines, ablations,
publication-grade statistics, saved artefacts and academic-style figures.

--------------------------------------------------------------------------------
PROTOCOL (what makes the numbers defensible)
--------------------------------------------------------------------------------
 1. Causality        Features at row t use data up to close(t) only. The signal
                     formed at close(t) is EXECUTED at close(t+EXEC_LAG) and held
                     H days:  y[t] = close[t+lag+H] / close[t+lag] - 1.
 2. Purging          A training label is used only if it is fully realised by the
                     time the test window starts (i + lag + H <= test_start).
 3. Validation       Early stopping / ridge alpha use a chronological VALIDATION
                     block cut from the END of the training window, purged from
                     the fit block. The test set is never touched for selection.
                     (RAN v3 early-stopped on the test fold; this fixes that.)
 4. Preprocessing    Feature mean/std and target scaling are fit on the training
                     window only; features are clipped to +/-5 sigma.
 5. Overlap          Labels overlap for H>1. Significance therefore uses
                     Newey-West (lag=H) t-statistics, a stationary block
                     bootstrap, and permutation tests on NON-overlapping rows.
 6. Costs            Turnover-based costs (bps of traded notional) + short borrow.
 7. Untouched test   --stage dev  : data after --holdout_start is never loaded.
                     --stage final: run ONCE after the config is frozen; results
                                    are reported for Full / Dev / Holdout.
 8. Multiple tests   Holm-adjusted one-sided IC p-values over the whole family of
                     (model x horizon); Probabilistic Sharpe Ratio; every trial
                     that was run is reported (nothing is dropped).
 9. Traceability     config.json, manifest.json (versions, data hash, git hash),
                     scores/*.npz (all OOS predictions), tables/*.csv|tex,
                     figures/*.pdf|png, results.json, run.log.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  pip install numpy pandas scipy matplotlib torch yfinance lxml

  # 1) development stage (holdout data never loaded) - tune here only
  # 0) verify yfinance download + universe filtering only (no training)
  python ran_research.py --check_data
  python ran_research.py --check_data --universe multiasset --start 2008-01-01

  python ran_research.py --stage dev   --out runs/dev

  # 2) frozen config, run once on everything
  python ran_research.py --stage final --out runs/final

  # smoke test on synthetic data (no internet, ridge/baselines only)
  python ran_research.py --synthetic 60 --models ridge,mom,rev,random \
      --horizons 10,20 --train_window 500 --n_boot 200 --n_perm 100 \
      --start 2008-01-01 --end 2022-12-31 --holdout_start 2020-01-01 --out runs/smoke

Survivorship note: the default universe is the CURRENT S&P 500 membership
(Wikipedia) restricted to names with full history -> survivorship-biased.
State this as a limitation in the paper, or supply a point-in-time
constituent file via  --universe csv --universe_csv my_universe.csv
(columns: ticker,sector[,industry]).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # RAN models unavailable; ridge/baselines still run
    torch = None

warnings.filterwarnings("ignore")
log = logging.getLogger("ran")


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    out: str = "runs/ran_main"
    data_cache: str = "data_cache"
    # universe / data
    universe: str = "sp500"          # sp500 | multiasset | default50 | csv
    universe_csv: str = ""
    max_assets: int = 0              # 0 = all
    synthetic: int = 0               # >0: synthetic universe of this size
    start: str = "2005-01-01"
    end: str = "2025-12-31"
    min_history_frac: float = 0.995  # min fraction of calendar a ticker must cover
    # model
    seq_len: int = 60
    knn_k: int = 10
    hidden: int = 32
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 15
    patience: int = 3
    n_seeds: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-3
    noise: float = 0.1
    ic_lambda: float = 2.0
    # experiment
    horizons: str = "10,20,60"
    models: str = "ran,ran_nograph,ridge,mom,rev,random"
    train_window: int = 1008
    expanding: bool = False
    step_days: int = 126
    min_train: int = 500
    val_frac: float = 0.2
    exec_lag: int = 1
    # portfolio / costs
    top_frac: float = 0.10
    cost_bps: float = 5.0            # per unit of traded notional
    borrow_bps: float = 50.0         # annual, on the short book
    cost_grid: str = "0,2.5,5,10,20"
    # protocol
    holdout_start: str = "2023-01-01"
    stage: str = "final"             # dev | final
    # statistics
    n_boot: int = 2000
    n_perm: int = 1000
    seed: int = 42
    device: str = "auto"
    resume: bool = True
    check_data: bool = False         # download + audit universe, then exit (no training)


MODEL_LABEL = {
    "ran": "RAN (CNN+GCN)", "ran_nograph": "RAN w/o graph", "ridge": "Ridge",
    "mom": "Momentum (120-20d)", "rev": "Reversal (5d)", "random": "Random",
}
MODEL_STYLE = {
    "ran": dict(color="#000000", ls="-", lw=1.5),
    "ran_nograph": dict(color="#0072B2", ls="--", lw=1.2),
    "ridge": dict(color="#D55E00", ls="-.", lw=1.2),
    "mom": dict(color="#009E73", ls=":", lw=1.3),
    "rev": dict(color="#CC79A7", ls=":", lw=1.3),
    "random": dict(color="#888888", ls="-", lw=0.8),
}
EPISODES = {
    "Euro crisis (Jul-Dec 2011)": ("2011-07-01", "2011-12-31"),
    "Aug-2015 / Feb-2016 stress": ("2015-08-01", "2016-02-29"),
    "Q4-2018 sell-off": ("2018-10-01", "2018-12-31"),
    "COVID crash (Feb-Apr 2020)": ("2020-02-15", "2020-04-30"),
    "2022 rate-shock bear market": ("2022-01-01", "2022-12-31"),
    "Regional-bank stress (Mar-May 2023)": ("2023-03-01", "2023-05-31"),
}


# =============================================================================
# 1. UNIVERSE + DATA
# =============================================================================
DEFAULT50 = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Comm", "AMZN": "ConsDisc", "TSLA": "ConsDisc",
    "JPM": "Fin", "WMT": "Stap", "JNJ": "Health", "V": "Fin", "MA": "Fin",
    "UNH": "Health", "HD": "ConsDisc", "BAC": "Fin", "DIS": "Comm", "ADBE": "Tech",
    "CRM": "Tech", "NFLX": "Comm", "INTC": "Tech", "CSCO": "Tech", "VZ": "Comm",
    "T": "Comm", "XOM": "Energy", "CVX": "Energy", "PFE": "Health", "MRK": "Health",
    "PEP": "Stap", "KO": "Stap", "COST": "Stap", "MCD": "ConsDisc", "NKE": "ConsDisc",
    "TMO": "Health", "DHR": "Health", "ABT": "Health", "LLY": "Health", "QCOM": "Tech",
    "TXN": "Tech", "AVGO": "Tech", "ORCL": "Tech", "IBM": "Tech", "ACN": "Tech",
    "MS": "Fin", "GS": "Fin", "AXP": "Fin", "SPGI": "Fin", "BLK": "Fin",
    "SCHW": "Fin", "C": "Fin", "MET": "Fin", "PRU": "Fin", "MMM": "Industrials",
}


# Multi-asset ETF universe (real traded instruments; asset class used for neutralisation groups).
# Mostly liquid US-listed ETFs with history from ~2007. Use --start 2008-01-01 (recommended).
MULTIASSET = {
    "US Equity": ["SPY", "QQQ", "IWM", "DIA", "MDY"],
    "US Sector": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"],
    "Intl Equity": ["EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ", "EWY", "EWT", "FXI", "EWA", "EWC", "EWH",
                    "EWS", "EWW", "EZU", "VGK"],
    "Fixed Income": ["TLT", "IEF", "SHY", "IEI", "TIP", "LQD", "AGG", "HYG", "MUB", "EMB", "BND"],
    "Commodity": ["GLD", "SLV", "USO", "DBC", "DBA", "UNG", "DBB"],
    "Currency": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC"],
    "Real Estate": ["VNQ", "IYR"],
}


def load_universe(cfg: Config) -> pd.DataFrame:
    cache = Path(cfg.data_cache)
    cache.mkdir(parents=True, exist_ok=True)
    if cfg.universe == "default50":
        df = pd.DataFrame({"ticker": list(DEFAULT50), "sector": list(DEFAULT50.values())})
        df["industry"] = df["sector"]
    elif cfg.universe == "multiasset":
        df = pd.DataFrame([(t, ac, ac) for ac, ts in MULTIASSET.items() for t in ts],
                          columns=["ticker", "sector", "industry"])
    elif cfg.universe == "csv":
        df = pd.read_csv(cfg.universe_csv)
        if "industry" not in df.columns:
            df["industry"] = df["sector"]
    else:  # sp500 via Wikipedia (cached)
        p = cache / "universe_sp500.csv"
        if p.exists():
            df = pd.read_csv(p)
        else:
            import urllib.request
            log.info("Fetching S&P 500 constituents from Wikipedia...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            try:
                html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
            except Exception as e:
                raise SystemExit(f"Could not fetch the S&P 500 list ({e}). Use --universe multiasset, "
                                 f"--universe default50, or --universe csv --universe_csv file.csv")
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
                               "sector": t["GICS Sector"], "industry": t["GICS Sub-Industry"]})
            df.to_csv(p, index=False)
    df = df.drop_duplicates("ticker").reset_index(drop=True)
    if cfg.max_assets and len(df) > cfg.max_assets:
        df = df.iloc[: cfg.max_assets].reset_index(drop=True)
    return df


def download_prices(tickers, cfg: Config) -> dict:
    import yfinance as yf
    cache = Path(cfg.data_cache)
    key = hashlib.sha1((",".join(sorted(tickers)) + cfg.start + cfg.end).encode()).hexdigest()[:10]
    path = cache / f"prices_{key}.pkl"
    if path.exists():
        log.info("Loading cached prices %s", path)
        return pd.read_pickle(path)
    out = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        log.info("Downloading %d-%d / %d", i, i + len(chunk), len(tickers))
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=cfg.start, end=cfg.end, group_by="ticker",
                                 auto_adjust=True, progress=False, threads=True)
                break
            except Exception as e:  # noqa
                log.warning("download retry %d: %s", attempt, e)
                time.sleep(3)
        if df is None:
            continue
        for t in chunk:
            try:
                d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
            except KeyError:
                continue
            if len(d) > 0:
                out[t] = d
    missing = [t for t in tickers if t not in out]
    if not out:
        raise SystemExit("yfinance returned no data. Check your internet connection / `pip install -U yfinance`.")
    if missing:
        log.warning("No data for %d tickers (delisted/renamed?): %s", len(missing), missing[:30])
    pd.to_pickle(out, path)
    return out


def make_synthetic(n_assets: int, cfg: Config):
    """Factor-model synthetic universe with a weak persistent alpha (for smoke tests)."""
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, cfg.end)
    T = len(dates)
    n_sec = 8
    sec = rng.integers(0, n_sec, n_assets)
    mkt = rng.normal(3e-4, 0.01, T)
    secf = rng.normal(0, 0.004, (T, n_sec))
    alpha = np.zeros((T, n_assets))
    innov = rng.normal(0, 0.0004, (T, n_assets))
    for t in range(1, T):
        alpha[t] = 0.98 * alpha[t - 1] + innov[t]
    beta = rng.uniform(0.6, 1.4, n_assets)
    ret = mkt[:, None] * beta + secf[:, sec] + alpha + rng.normal(0, 0.012, (T, n_assets))
    close = 50 * np.exp(np.cumsum(ret, axis=0))
    prices = {}
    tick = [f"S{i:03d}" for i in range(n_assets)]
    for j, t in enumerate(tick):
        c = close[:, j]
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, T))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, T)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, T)))
        v = np.exp(rng.normal(13, 0.5, T))
        prices[t] = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=dates)
    uni = pd.DataFrame({"ticker": tick, "sector": [f"Sec{s}" for s in sec], "industry": [f"Sec{s}" for s in sec]})
    return prices, uni


def build_master(prices: dict, uni: pd.DataFrame, cfg: Config):
    """Common calendar, strict full-history filter. Returns aligned OHLCV frames + audit table."""
    closes = pd.concat({t: d["Close"] for t, d in prices.items()}, axis=1)
    cal = closes.index[closes.notna().mean(axis=1) >= 0.5]
    audit = []
    keep = []
    for t in prices:
        c = prices[t]["Close"].reindex(cal).ffill(limit=3)
        cov = float(c.notna().mean())
        first_ok = c.first_valid_index() is not None and c.index.get_loc(c.first_valid_index()) <= 10
        if cov >= cfg.min_history_frac and first_ok and (c.dropna() > 0).all():
            keep.append(t)
            audit.append((t, "kept", cov))
        else:
            audit.append((t, "dropped: insufficient history/quality", cov))
    if len(keep) < 20:
        raise RuntimeError(f"Only {len(keep)} tickers survived the history filter; "
                           f"lower --min_history_frac or move --start later.")
    fields = {}
    for fld in ["Open", "High", "Low", "Close", "Volume"]:
        fr = pd.concat({t: prices[t][fld].reindex(cal) for t in keep}, axis=1)
        fr = fr.ffill(limit=3).bfill(limit=10) if fld != "Volume" else fr.fillna(0.0)
        fields[fld] = fr
    for fld in ["Open", "High", "Low"]:
        fields[fld] = fields[fld].fillna(fields["Close"])
    ok = fields["Close"].notna().all(axis=0)
    keep = [t for t in keep if ok[t]]
    for fld in fields:
        fields[fld] = fields[fld][keep]
    sec = uni.set_index("ticker").reindex(keep)
    audit_df = pd.DataFrame(audit, columns=["ticker", "status", "coverage"])
    return fields, sec, audit_df


# =============================================================================
# 2. FEATURES (causal)
# =============================================================================
def per_asset_features(o, h, l, c, v) -> pd.DataFrame:
    f = pd.DataFrame(index=c.index)
    for lag in [1, 3, 5, 10, 20]:
        f[f"ret_{lag}"] = c.pct_change(lag)
        f[f"high_ret_{lag}"] = h.pct_change(lag)
        f[f"low_ret_{lag}"] = l.pct_change(lag)
    for w in (5, 20, 60):
        f[f"vol_{w}"] = f["ret_1"].rolling(w).std()
    diff = c.diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-6))
    vol_mean = v.rolling(5).mean() + 1e-6
    f["ofi"] = (c.diff() * v.diff()).rolling(5).mean() / vol_mean
    f["mom_60_5"] = f["ret_1"].rolling(60).sum() - f["ret_1"].rolling(5).sum()
    f["mom_120_20"] = f["ret_1"].rolling(120).sum() - f["ret_1"].rolling(20).sum()
    f["overnight"] = o / c.shift(1) - 1
    f["intraday"] = c / o - 1
    f["ov_20"] = f["overnight"].rolling(20).mean()
    f["id_20"] = f["intraday"].rolling(20).mean()
    dv = (c * v).replace(0, np.nan)
    f["amihud"] = (f["ret_1"].abs() / dv).rolling(20).mean()
    return f.replace([np.inf, -np.inf], np.nan)


@dataclass
class Panel:
    dates: np.ndarray            # (T,) datetime64[D]
    tickers: list
    sectors: np.ndarray          # (A,) str
    sector_ids: np.ndarray       # (A,) int
    F: np.ndarray                # (T, A, Ff) float32 raw features (may contain NaN)
    fnames: list
    ret: np.ndarray              # (T, A) daily returns
    close: np.ndarray            # (T, A)
    knn_idx: np.ndarray          # (T, A, K) int16
    knn_w: np.ndarray            # (T, A, K) float32
    s0: int                      # first valid sample row
    data_hash: str = ""


def build_knn(ret: np.ndarray, seq: int, k: int):
    """Correlation kNN graph at each date from the trailing `seq` returns (causal)."""
    T, A = ret.shape
    idx = np.zeros((T, A, k), np.int16)
    w = np.zeros((T, A, k), np.float32)
    r = np.nan_to_num(ret, nan=0.0).astype(np.float32)
    t0 = time.time()
    for t in range(seq, T):
        win = r[t - seq + 1: t + 1]
        z = (win - win.mean(0)) / (win.std(0) + 1e-8)
        corr = (z.T @ z) / seq
        np.fill_diagonal(corr, -np.inf)
        nn_idx = np.argpartition(corr, -k, axis=1)[:, -k:]
        idx[t] = nn_idx
        w[t] = np.clip(np.take_along_axis(corr, nn_idx, 1), 0.0, None)
        if t % 1000 == 0:
            log.info("  kNN graph %d/%d  (%.0fs)", t, T, time.time() - t0)
    return idx, w


def build_panel(fields: dict, sec: pd.DataFrame, cfg: Config) -> Panel:
    O, H, L, C, V = (fields[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    tickers = list(C.columns)
    idx = C.index
    log.info("Computing per-asset features for %d tickers...", len(tickers))
    feats = {t: per_asset_features(O[t], H[t], L[t], C[t], V[t]) for t in tickers}
    base_cols = list(feats[tickers[0]].columns)
    B = np.stack([feats[t][base_cols].values for t in tickers], axis=1).astype(np.float32)  # (T,A,Fb)
    ret_df = C.pct_change()
    mkt = ret_df.mean(axis=1)

    cs_j = [j for j, c in enumerate(base_cols) if not c.startswith("ret_")]
    sub = B[:, :, cs_j]
    mu = np.nanmean(sub, axis=1, keepdims=True)
    sd = np.nanstd(sub, axis=1, keepdims=True) + 1e-9
    Z = (sub - mu) / sd
    rank_cols = [base_cols.index(c) for c in ["ret_5", "ret_20", "mom_60_5", "mom_120_20"]]
    R = np.stack([pd.DataFrame(B[:, :, j]).rank(axis=1, pct=True).values for j in rank_cols], axis=2)
    EX = np.stack([(ret_df.rolling(lag).sum().sub(mkt.rolling(lag).sum(), axis=0)).values
                   for lag in (5, 20, 60)], axis=2)
    beta = (ret_df.rolling(60).cov(mkt)).div(mkt.rolling(60).var() + 1e-9, axis=0).values[:, :, None]
    Fall = np.concatenate([B, Z, R, EX, beta], axis=2).astype(np.float32)
    Fall[~np.isfinite(Fall)] = np.nan
    fnames = (base_cols + [f"cs_z_{base_cols[j]}" for j in cs_j] +
              [f"cs_rank_{base_cols[j]}" for j in rank_cols] +
              [f"excess_mom_{l}" for l in (5, 20, 60)] + ["beta_60"])

    good = np.isfinite(Fall).mean(axis=(1, 2)) >= 0.995
    r0 = int(np.argmax(good))
    log.info("Warm-up cut: dropping first %d rows", r0)
    sl = slice(r0, None)
    Fall, ret, close, dates = Fall[sl], ret_df.values[sl].astype(np.float32), C.values[sl].astype(np.float32), idx.values[sl]
    knn_i, knn_w = build_knn(ret, cfg.seq_len, cfg.knn_k)
    sectors = sec["sector"].fillna("Unknown").values.astype(str)
    _, sector_ids = np.unique(sectors, return_inverse=True)
    dh = hashlib.sha256(close.tobytes()).hexdigest()[:16]
    return Panel(np.array(dates, dtype="datetime64[D]"), tickers, sectors, sector_ids, Fall, fnames,
                 ret, close, knn_i, knn_w, s0=cfg.seq_len, data_hash=dh)


def truncate_panel(p: Panel, end_excl: np.datetime64) -> Panel:
    n = int(np.searchsorted(p.dates, end_excl))
    return dataclasses.replace(p, dates=p.dates[:n], F=p.F[:n], ret=p.ret[:n], close=p.close[:n],
                               knn_idx=p.knn_idx[:n], knn_w=p.knn_w[:n])


def make_target(close: np.ndarray, h: int, lag: int) -> np.ndarray:
    """y[t] = close[t+lag+h] / close[t+lag] - 1 ; NaN when not yet realised."""
    T = len(close)
    y = np.full_like(close, np.nan, dtype=np.float32)
    n = T - lag - h
    if n > 0:
        y[:n] = close[lag + h:] / close[lag: lag + n] - 1.0
    return y


# =============================================================================
# 3. MODEL (torch)
# =============================================================================
if torch is not None:
    class TemporalEncoder(nn.Module):
        def __init__(self, n_features, hidden):
            super().__init__()
            self.c1 = nn.Conv1d(n_features, hidden, 3, padding=2, dilation=1)
            self.c2 = nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2)
            self.c3 = nn.Conv1d(hidden, hidden, 3, padding=8, dilation=4)

        def forward(self, x):                       # x: (B, A, T, F)
            B, A, T, Fd = x.shape
            x = x.reshape(B * A, T, Fd).permute(0, 2, 1)
            for conv, trim in ((self.c1, 2), (self.c2, 4), (self.c3, 8)):
                x = F.relu(conv(x))[:, :, :-trim]   # trim right padding -> causal
            return x.mean(dim=-1).view(B, A, -1)

    class GraphConv(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x, adj):
            B, A, _ = x.shape
            a = adj + torch.eye(A, device=x.device, dtype=x.dtype).unsqueeze(0)
            a = a / a.sum(-1, keepdim=True).clamp(min=1e-6)
            return F.relu(self.lin(torch.bmm(a, x))) + x

    class SmallRAN(nn.Module):
        def __init__(self, n_features, hidden, dropout, use_graph=True):
            super().__init__()
            self.use_graph = use_graph
            self.enc = TemporalEncoder(n_features, hidden)
            self.gcn = GraphConv(hidden)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x, adj=None):
            h = self.enc(x)
            if self.use_graph:
                h = self.gcn(h, adj)
            return self.fc(self.drop(h)).squeeze(-1)

    def ic_loss(s, t):
        s = s - s.mean(1, keepdim=True)
        t = t - t.mean(1, keepdim=True)
        return 1.0 - ((s * t).sum(1) / (s.norm(dim=1) * t.norm(dim=1) + 1e-8)).mean()

    class TorchEngine:
        """Holds the panel on-device; builds normalised windows + dense adjacency lazily."""

        def __init__(self, panel: Panel, cfg: Config, device):
            self.cfg, self.dev = cfg, device
            self.Fraw = torch.from_numpy(panel.F).to(device)
            self.knn_idx = torch.from_numpy(panel.knn_idx).to(device)
            self.knn_w = torch.from_numpy(panel.knn_w).to(device)
            self.offs = torch.arange(-cfg.seq_len + 1, 1, device=device)
            self.Fn = None

        def set_norm(self, mu, sd):
            m = torch.from_numpy(mu).to(self.dev)
            s = torch.from_numpy(sd).to(self.dev)
            self.Fn = None
            self.Fn = torch.nan_to_num(((self.Fraw - m) / s).clamp(-5, 5), nan=0.0)

        def gather(self, idx, use_graph):
            it = torch.as_tensor(idx, device=self.dev, dtype=torch.long)
            x = self.Fn[it[:, None] + self.offs[None, :]].permute(0, 2, 1, 3).contiguous()  # (B,A,T,F)
            adj = None
            if use_graph:
                ki = self.knn_idx[it].long()
                kw = self.knn_w[it]
                B, A, _ = ki.shape
                adj = torch.zeros(B, A, A, device=self.dev)
                adj.scatter_(2, ki, kw)
                adj = 0.5 * (adj + adj.transpose(1, 2))
                rs = adj.sum(-1, keepdim=True)
                adj = adj / torch.where(rs == 0, torch.ones_like(rs), rs)
            return x, adj

        @torch.no_grad()
        def predict(self, model, idx, use_graph, batch=32):
            model.eval()
            out = []
            for i in range(0, len(idx), batch):
                x, adj = self.gather(idx[i:i + batch], use_graph)
                out.append(model(x, adj).float().cpu().numpy())
            return np.concatenate(out)

    def train_ran(eng, fit_idx, val_idx, y_fit_std, y_val_raw, seed, use_graph, cfg):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        A = y_fit_std.shape[1]
        model = SmallRAN(eng.Fn.shape[-1], cfg.hidden, cfg.dropout, use_graph).to(eng.dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        ytr = torch.from_numpy(y_fit_std).to(eng.dev)
        best_ic, best_state, best_ep, wait = -np.inf, None, 0, 0
        for ep in range(cfg.epochs):
            model.train()
            perm = rng.permutation(len(fit_idx))
            for b in range(0, len(perm) - cfg.batch_size + 1, cfg.batch_size):
                bi = perm[b:b + cfg.batch_size]
                x, adj = eng.gather(fit_idx[bi], use_graph)
                x = x + cfg.noise * torch.randn_like(x)
                pred = model(x, adj)
                yb = ytr[torch.as_tensor(bi, device=eng.dev)]
                loss = F.mse_loss(pred, yb) + cfg.ic_lambda * ic_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            va_ic = float(np.mean(daily_rank_ic(eng.predict(model, val_idx, use_graph), y_val_raw)))
            if va_ic > best_ic:
                best_ic, best_ep, wait = va_ic, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_ic, best_ep


# =============================================================================
# 4. WALK-FORWARD PLAN (identical for every model -> paired comparisons valid)
# =============================================================================
def cs_standardize(y):
    return ((y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-6)).astype(np.float32)


def plan_windows(panel: Panel, y: np.ndarray, h: int, cfg: Config):
    T = len(panel.dates)
    span = cfg.exec_lag + h
    plan = []
    ts = panel.s0 + cfg.train_window
    while ts < T:
        te = min(ts + cfg.step_days, T)
        lo = panel.s0 if cfg.expanding else max(panel.s0, ts - cfg.train_window)
        cand = np.arange(lo, ts - span + 1)                      # label realised by ts
        cand = cand[~np.isnan(y[cand]).any(axis=1)]
        test = np.arange(ts, te)
        test = test[~np.isnan(y[test]).any(axis=1)]
        n_val = int(len(cand) * cfg.val_frac)
        if n_val >= 20 and len(test) > 0:
            val = cand[-n_val:]
            fit = cand[cand <= val[0] - span]                    # purge fit labels that reach into val
            if len(fit) >= cfg.min_train:
                block = panel.F[max(0, lo - cfg.seq_len + 1): cand[-1] + 1]
                mu = np.nanmean(block, axis=(0, 1)).astype(np.float32)
                sd = (np.nanstd(block, axis=(0, 1)) + 1e-6).astype(np.float32)
                plan.append(dict(ts=ts, te=te, fit=fit, val=val, test=test, mu=mu, sd=sd))
        ts = te
    return plan


def frozen_plan(plan_full, panel, y, h, cfg):
    """Single frozen model trained before holdout_start, applied to the whole holdout."""
    hs = int(np.searchsorted(panel.dates, np.datetime64(cfg.holdout_start)))
    c2 = dataclasses.replace(cfg, step_days=len(panel.dates))
    T = len(panel.dates)
    span = cfg.exec_lag + h
    lo = max(panel.s0, hs - cfg.train_window)
    cand = np.arange(lo, hs - span + 1)
    cand = cand[~np.isnan(y[cand]).any(axis=1)]
    test = np.arange(hs, T)
    test = test[~np.isnan(y[test]).any(axis=1)]
    n_val = int(len(cand) * cfg.val_frac)
    val = cand[-n_val:]
    fit = cand[cand <= val[0] - span]
    block = panel.F[max(0, lo - cfg.seq_len + 1): cand[-1] + 1]
    mu = np.nanmean(block, axis=(0, 1)).astype(np.float32)
    sd = (np.nanstd(block, axis=(0, 1)) + 1e-6).astype(np.float32)
    return [dict(ts=hs, te=T, fit=fit, val=val, test=test, mu=mu, sd=sd)]


def norm_rows(F_np, idx, mu, sd):
    x = (F_np[idx] - mu) / sd
    return np.nan_to_num(np.clip(x, -5, 5), nan=0.0)


def ridge_window(panel, w, y, seed_unused=0):
    Xf = norm_rows(panel.F, w["fit"], w["mu"], w["sd"])            # (n,A,F)
    Xv = norm_rows(panel.F, w["val"], w["mu"], w["sd"])
    Xt = norm_rows(panel.F, w["test"], w["mu"], w["sd"])
    yf = cs_standardize(y[w["fit"]])
    yv = cs_standardize(y[w["val"]])
    D = Xf.shape[-1]

    def fit(X, Y, alpha):
        Xm = X.reshape(-1, D)
        yy = Y.reshape(-1)
        xm, ym = Xm.mean(0), yy.mean()
        Xc = Xm - xm
        beta = np.linalg.solve(Xc.T @ Xc + alpha * len(Xc) * np.eye(D), Xc.T @ (yy - ym))
        return beta

    best, best_a = -np.inf, 1.0
    for a in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        b = fit(Xf, yf, a)
        ic = float(np.mean(daily_rank_ic(Xv @ b, y[w["val"]])))
        if ic > best:
            best, best_a = ic, a
    Xall = np.concatenate([Xf, Xv])
    yall = np.concatenate([yf, yv])
    b = fit(Xall, yall, best_a)
    return (Xt @ b).astype(np.float32), dict(val_ic=best, alpha=best_a)


def run_model(kind, panel, y, plan, cfg, eng=None, device=None):
    T, A = y.shape
    S = np.full((T, A), np.nan, np.float32)
    diags = []
    col = {n: i for i, n in enumerate(panel.fnames)}
    rng = np.random.default_rng(cfg.seed)
    for wi, w in enumerate(plan):
        t0 = time.time()
        if kind in ("ran", "ran_nograph"):
            use_graph = kind == "ran"
            eng.set_norm(w["mu"], w["sd"])
            yf = cs_standardize(y[w["fit"]])
            preds, vics, eps = [], [], []
            for s in range(cfg.n_seeds):
                m, vic, ep = train_ran(eng, w["fit"], w["val"], yf, y[w["val"]], cfg.seed + s, use_graph, cfg)
                preds.append(eng.predict(m, w["test"], use_graph))
                vics.append(vic); eps.append(ep)
                del m
            S[w["test"]] = np.mean(preds, axis=0)
            d = dict(val_ic=float(np.mean(vics)), best_epoch=float(np.mean(eps)))
        elif kind == "ridge":
            S[w["test"]], d = ridge_window(panel, w, y)
        elif kind == "mom":
            S[w["test"]] = panel.F[w["test"], :, col["mom_120_20"]]
            d = {}
        elif kind == "rev":
            S[w["test"]] = -panel.F[w["test"], :, col["ret_5"]]
            d = {}
        elif kind == "random":
            S[w["test"]] = rng.standard_normal((len(w["test"]), A)).astype(np.float32)
            d = {}
        else:
            raise ValueError(kind)
        S[w["test"]] = np.nan_to_num(S[w["test"]], nan=0.0)
        d.update(window=wi, test_start=str(panel.dates[w["ts"]]), test_end=str(panel.dates[w["te"] - 1]),
                 n_fit=len(w["fit"]), n_val=len(w["val"]), n_test=len(w["test"]), sec=round(time.time() - t0, 1))
        diags.append(d)
        log.info("  [%s] window %d/%d  %s -> %s  fit=%d  %s  (%.1fs)", kind, wi + 1, len(plan), d["test_start"],
                 d["test_end"], d["n_fit"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                                             if k in ("val_ic", "best_epoch", "alpha")}, d["sec"])
    return S, diags


# =============================================================================
# 5. METRICS & STATISTICS
# =============================================================================
def daily_rank_ic(S, Y):
    rs = stats.rankdata(S, axis=1)
    ry = stats.rankdata(Y, axis=1)
    a = rs - rs.mean(1, keepdims=True)
    b = ry - ry.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12)


def sector_neutralize(S, sector_ids, min_size=3):
    out = S.copy()
    for s in np.unique(sector_ids):
        m = sector_ids == s
        if m.sum() >= min_size:
            out[:, m] = S[:, m] - S[:, m].mean(1, keepdims=True)
    return out


def valid_rows(S, Y):
    return ~np.isnan(S).any(1) & ~np.isnan(Y).any(1)


def nw_tstat(x, lag):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return np.nan
    xm = x - x.mean()
    g = xm @ xm / n
    for l in range(1, min(lag, n - 1) + 1):
        g += 2 * (1 - l / (lag + 1)) * (xm[l:] @ xm[:-l] / n)
    se = np.sqrt(max(g, 1e-18) / n)
    return x.mean() / se


def sharpe(r, h):
    r = np.asarray(r)
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / s * np.sqrt(252.0 / h)) if s > 1e-12 else 0.0


def max_drawdown(eq):
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0


def strategy_returns(S, Y, h, frac, cost_bps, borrow_bps):
    """Staggered book: a new tranche is opened every day (one per row), each held H days.
    Long top-`frac`, short bottom-`frac`, gross exposure 1 (0.5 long / 0.5 short)."""
    T, A = S.shape
    vi = np.flatnonzero(valid_rows(S, Y))
    k = max(1, int(round(frac * A)))
    order = np.argsort(S[vi], axis=1)
    R = np.arange(len(vi))[:, None]
    Wv = np.zeros((len(vi), A), np.float32)
    Wv[R, order[:, :k]] = -0.5 / k
    Wv[R, order[:, -k:]] = 0.5 / k
    W = np.zeros((T, A), np.float32)
    W[vi] = Wv
    gross = (Wv * Y[vi]).sum(1).astype(np.float64)
    m = vi - h
    prev = np.zeros_like(Wv)
    has = m >= 0
    prev[has] = W[m[has]]
    turn = np.abs(Wv - prev).sum(1)
    borrow = 0.5 * borrow_bps * 1e-4 * h / 252.0
    net = gross - cost_bps * 1e-4 * turn - borrow
    steady = has & valid_rows(S, Y)[np.clip(m, 0, T - 1)]
    return dict(rows=vi, gross=gross, net=net, turn=turn, turnover=float(turn[steady].mean() / 2) if steady.any() else np.nan)


class BootCache:
    def __init__(self, B, seed):
        self.B, self.seed, self.c = B, seed, {}

    def get(self, n, L):
        key = (n, int(L))
        if key not in self.c:
            rng = np.random.default_rng(self.seed + n * 7 + int(L))
            p = 1.0 / max(L, 1.0)
            idx = np.empty((self.B, n), np.int32)
            idx[:, 0] = rng.integers(0, n, self.B)
            new = rng.random((self.B, n)) < p
            st = rng.integers(0, n, (self.B, n))
            for t in range(1, n):
                idx[:, t] = np.where(new[:, t], st[:, t], (idx[:, t - 1] + 1) % n)
            self.c = {key: idx}   # keep only the latest (memory)
        return self.c[key]


def boot_sharpe(r, h, idx):
    x = r[idx]
    return x.mean(1) / np.maximum(x.std(1, ddof=1), 1e-12) * np.sqrt(252.0 / h)


def psr(r_nonoverlap, sr_star=0.0):
    r = np.asarray(r_nonoverlap)
    n = len(r)
    if n < 10 or r.std(ddof=1) < 1e-12:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-9))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / den))


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def permutation_test(S, Y, h, frac, n_perm, seed, n_offsets=3):
    """Null: shuffle returns across assets within each day (non-overlapping rows only).
    Statistic: gross annualised Sharpe of the long-short book."""
    vi = np.flatnonzero(valid_rows(S, Y))
    A = S.shape[1]
    k = max(1, int(round(frac * A)))
    offs = np.unique(np.linspace(0, h - 1, n_offsets).astype(int))
    pvals, nulls, actuals = [], None, []
    for o in offs:
        rows = vi[o::h]
        if len(rows) < 12:
            continue
        Ss, Ys = S[rows], Y[rows]
        order = np.argsort(Ss, axis=1)
        R = np.arange(len(rows))[:, None]
        act = sharpe(0.5 * (Ys[R, order[:, -k:]].mean(1) - Ys[R, order[:, :k]].mean(1)), h)
        rng = np.random.default_rng(seed + int(o))
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.random(Ys.shape).argsort(1)
            g = 0.5 * (Ys[R, perm[:, :k]].mean(1) - Ys[R, perm[:, k:2 * k]].mean(1))
            null[i] = sharpe(g, h)
        pvals.append((1 + (null >= act).sum()) / (1 + n_perm))
        actuals.append(act)
        if nulls is None:
            nulls = null
    return dict(p_median=float(np.median(pvals)) if pvals else np.nan, p_all=[float(p) for p in pvals],
                actual=float(np.median(actuals)) if actuals else np.nan, null=nulls)


def analyse(S, Y, h, panel, cfg, period_mask, variant, boot: BootCache):
    Su = S.copy()
    Su[~period_mask] = np.nan
    if variant == "sector":
        Su = sector_neutralize(Su, panel.sector_ids)
    vi = np.flatnonzero(valid_rows(Su, Y))
    if len(vi) < 30:
        return None
    ic = daily_rank_ic(Su[vi], Y[vi])
    t_nw = nw_tstat(ic, h)
    res = dict(n_rows=len(vi), n_indep=len(vi) / h, ic=float(ic.mean()), ic_std=float(ic.std()),
               icir=float(ic.mean() / (ic.std() + 1e-12)), ic_t_nw=float(t_nw),
               ic_p_nw=float(1 - stats.norm.cdf(t_nw)))            # one-sided, H1: IC > 0
    st = strategy_returns(Su, Y, h, cfg.top_frac, cfg.cost_bps, cfg.borrow_bps)
    net, gross = st["net"], st["gross"]
    eq = np.cumprod(1 + net / h)
    res.update(sharpe_net=sharpe(net, h), sharpe_gross=sharpe(gross, h), ann_ret=float(net.mean() * 252 / h),
               ann_vol=float(net.std(ddof=1) * np.sqrt(252 / h)), max_dd=max_drawdown(eq),
               turnover=st["turnover"], hit=float((net > 0).mean()))
    L = max(2 * h, 5)
    idx = boot.get(len(net), L)
    bs = boot_sharpe(net, h, idx)
    res.update(sharpe_lo=float(np.percentile(bs, 2.5)), sharpe_hi=float(np.percentile(bs, 97.5)),
               p_sharpe_le0=float((bs <= 0).mean()), psr=psr(net[::h]))
    res["_ic"] = ic
    res["_net"] = net
    res["_gross"] = gross
    res["_rows"] = st["rows"]
    res["_dates"] = panel.dates[vi]
    res["_eq"] = eq
    res["_S"] = Su
    return res


def cost_sensitivity(Su, Y, h, cfg, grid):
    out = {}
    for c in grid:
        st = strategy_returns(Su, Y, h, cfg.top_frac, c, cfg.borrow_bps)
        out[c] = sharpe(st["net"], h)
    return out


def decile_profile(S, Y):
    vi = np.flatnonzero(valid_rows(S, Y))
    A = S.shape[1]
    rk = stats.rankdata(S[vi], axis=1, method="ordinal") - 1
    dec = (rk * 10 // A).astype(int)
    prof = np.zeros((len(vi), 10))
    for d in range(10):
        m = dec == d
        prof[:, d] = np.where(m, Y[vi], 0).sum(1) / np.maximum(m.sum(1), 1)
    return prof


# =============================================================================
# 6. ORCHESTRATION
# =============================================================================
def fingerprint(cfg: Config, data_hash: str, h: int, kind: str) -> str:
    d = dataclasses.asdict(cfg)
    for k in ("out", "resume", "n_boot", "n_perm", "cost_grid", "cost_bps", "borrow_bps", "top_frac",
              "models", "horizons", "device"):
        d.pop(k, None)
    return hashlib.sha1((json.dumps(d, sort_keys=True) + data_hash + str(h) + kind).encode()).hexdigest()[:12]


def get_scores(kind, h, panel, y, plan, cfg, out, eng):
    fp = fingerprint(cfg, panel.data_hash, h, kind)
    path = out / "scores" / f"{kind}_h{h}.npz"
    if cfg.resume and path.exists():
        z = np.load(path, allow_pickle=True)
        if str(z["fp"]) == fp:
            log.info("[%s h=%d] loaded cached scores", kind, h)
            return z["scores"]
    log.info("[%s h=%d] running %d windows", kind, h, len(plan))
    S, diags = run_model(kind, panel, y, plan, cfg, eng)
    np.savez_compressed(path, scores=S, dates=panel.dates.astype("datetime64[D]").astype(str),
                        tickers=np.array(panel.tickers), fp=fp)
    pd.DataFrame(diags).to_csv(out / "diagnostics" / f"windows_{kind}_h{h}.csv", index=False)
    return S


def periods_for(cfg, panel):
    hs = np.datetime64(cfg.holdout_start)
    P = {"full": np.ones(len(panel.dates), bool)}
    if cfg.stage == "final":
        P["dev"] = panel.dates < hs
        P["holdout"] = panel.dates >= hs
    return P


# ----------------------------- tables -------------------------------------
def _esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def df_to_latex(df, path, caption, label, fmt="{:.3f}"):
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else fmt.format(v)
        return _esc(v)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{_esc(caption)}}}", rf"\label{{{label}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", r"\toprule",
         " & ".join(_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(cell(r[c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(path).write_text("\n".join(L))


def save_table(df, out, name, caption):
    df.to_csv(out / "tables" / f"{name}.csv", index=False)
    df_to_latex(df, out / "tables" / f"{name}.tex", caption, f"tab:{name}")


# ----------------------------- figures ------------------------------------
def apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4,
        "grid.linestyle": "-", "axes.axisbelow": True, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def savefig(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}")
    plt.close(fig)


def _panel_label(ax, i, txt):
    ax.set_title(f"({chr(97 + i)}) {txt}", loc="left", fontsize=9)


def _shade_holdout(ax, cfg):
    if cfg.stage == "final":
        import matplotlib.dates as mdates
        x0 = mdates.date2num(pd.Timestamp(cfg.holdout_start).to_pydatetime())
        x1 = ax.get_xlim()[1]
        if x1 > x0:
            ax.axvspan(x0, x1, color="#000000", alpha=0.07, lw=0)
            ax.set_xlim(right=x1)


def fig_equity(R, panel, cfg, models, horizons, out):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.9), squeeze=False)
    ew_daily = (panel.close[1:] / panel.close[:-1] - 1).mean(1)
    ew_daily = np.r_[0.0, ew_daily]
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        for m in models:
            r = R.get(("full", m, h, "raw"))
            if r is None or m == "random":
                continue
            ax.plot(pd.to_datetime(r["_dates"]), r["_eq"] - 1, label=MODEL_LABEL[m], **MODEL_STYLE[m])
        ref = R.get(("full", models[0], h, "raw"))
        if ref is not None:
            rows = ref["_rows"]
            ax.plot(pd.to_datetime(ref["_dates"]), np.cumprod(1 + ew_daily[rows]) - 1, color="#777777",
                    lw=0.9, ls=(0, (5, 2)), label="EW universe (long-only)")
        ax.axhline(0, color="k", lw=0.5)
        _shade_holdout(ax, cfg)
        _panel_label(ax, i, rf"$H={h}$ days")
        ax.set_ylabel("Cumulative net return" if i == 0 else "")
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    savefig(fig, out, "fig1_cumulative_net_return")


def fig_rolling_ic(R, cfg, models, horizons, out):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.7), squeeze=False)
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        for m in [x for x in models if x in ("ran", "ran_nograph", "ridge")]:
            r = R.get(("full", m, h, "raw"))
            if r is None:
                continue
            s = pd.Series(r["_ic"], index=pd.to_datetime(r["_dates"])).rolling(252, min_periods=126).mean()
            ax.plot(s.index, s.values, label=MODEL_LABEL[m], **MODEL_STYLE[m])
        ax.axhline(0, color="k", lw=0.5)
        _shade_holdout(ax, cfg)
        _panel_label(ax, i, rf"$H={h}$ days")
        ax.set_ylabel("Rolling 252-day mean rank IC" if i == 0 else "")
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    savefig(fig, out, "fig2_rolling_rank_ic")


def fig_sharpe_forest(R, models, horizons, period, out):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.9), squeeze=False, sharex=True)
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        ys, k = [], 0
        for m in models:
            r = R.get((period, m, h, "raw"))
            if r is None:
                continue
            c = MODEL_STYLE[m]["color"]
            ax.errorbar(r["sharpe_net"], k, xerr=[[r["sharpe_net"] - r["sharpe_lo"]], [r["sharpe_hi"] - r["sharpe_net"]]],
                        fmt="o", color=c, ms=4, capsize=2.5, lw=1.0)
            ys.append(MODEL_LABEL[m]); k += 1
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys if i == 0 else [""] * len(ys))
        ax.invert_yaxis()
        ax.set_xlabel("Net Sharpe ratio (95% bootstrap CI)")
        _panel_label(ax, i, rf"$H={h}$ days")
    fig.tight_layout()
    savefig(fig, out, f"fig3_sharpe_forest_{period}")


def fig_cost(cost_res, models, horizons, grid, out):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.7), squeeze=False, sharey=True)
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        for m in models:
            c = cost_res.get((m, h))
            if c is None or m == "random":
                continue
            ax.plot(grid, [c[g] for g in grid], marker="o", ms=3, label=MODEL_LABEL[m], **MODEL_STYLE[m])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("One-way cost (bps)")
        ax.set_ylabel("Net Sharpe ratio" if i == 0 else "")
        _panel_label(ax, i, rf"$H={h}$ days")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    savefig(fig, out, "fig4_cost_sensitivity")


def fig_annual_ic(R, models, horizons, out):
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    m = "ran" if any(k[1] == "ran" for k in R) else models[0]
    grays = ["#222222", "#777777", "#BBBBBB", "#DDDDDD"]
    years = None
    for j, h in enumerate(horizons):
        r = R.get(("full", m, h, "raw"))
        if r is None:
            continue
        s = pd.Series(r["_ic"], index=pd.to_datetime(r["_dates"])).groupby(lambda d: d.year).mean()
        years = s.index if years is None else years.union(s.index)
        w = 0.8 / len(horizons)
        ax.bar(np.arange(len(s)) + j * w, s.values, w, label=rf"$H={h}$", color=grays[j % 4], edgecolor="k", lw=0.4)
        ax.set_xticks(np.arange(len(s)) + 0.4 - w / 2)
        ax.set_xticklabels(s.index, rotation=45)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Mean daily rank IC")
    ax.set_title(f"Annual mean rank IC - {MODEL_LABEL[m]}", loc="left")
    ax.legend(ncol=len(horizons))
    fig.tight_layout()
    savefig(fig, out, "fig5_annual_rank_ic")


def fig_perm(perm, horizons, out, model="ran"):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.6), squeeze=False)
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        p = perm.get((model, h))
        if p is None or p["null"] is None:
            ax.axis("off"); continue
        ax.hist(p["null"], bins=35, color="#BBBBBB", edgecolor="#555555", lw=0.3)
        ax.axvline(p["actual"], color="k", lw=1.4)
        ax.set_xlabel("Gross Sharpe (non-overlapping rows)")
        ax.set_ylabel("Frequency" if i == 0 else "")
        ax.text(0.97, 0.93, rf"$p={p['p_median']:.3f}$", transform=ax.transAxes, ha="right", va="top")
        _panel_label(ax, i, rf"$H={h}$ days")
    fig.tight_layout()
    savefig(fig, out, f"fig6_permutation_null_{model}")


def fig_deciles(R, horizons, out, model="ran"):
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 0.6, 2.6), squeeze=False, sharey=False)
    for i, h in enumerate(horizons):
        ax = axes[0, i]
        r = R.get(("full", model, h, "raw"))
        if r is None:
            ax.axis("off"); continue
        prof = r["_dec"]
        m = prof.mean(0) * 1e4 / 1
        se = prof.std(0, ddof=1) / np.sqrt(len(prof) / h) * 1e4
        ax.bar(np.arange(1, 11), m, yerr=1.96 * se, color="#999999", edgecolor="k", lw=0.4, capsize=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Predicted-score decile (1 = lowest)")
        ax.set_ylabel(r"Mean forward return (bp)" if i == 0 else "")
        _panel_label(ax, i, rf"$H={h}$ days")
    fig.tight_layout()
    savefig(fig, out, f"fig7_decile_returns_{model}")


# ----------------------------- main run -----------------------------------
def env_manifest(cfg, panel):
    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:
            return None
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = None
    return dict(python=sys.version, platform=platform.platform(), numpy=ver("numpy"), pandas=ver("pandas"),
                scipy=ver("scipy"), torch=ver("torch"), yfinance=ver("yfinance"), matplotlib=ver("matplotlib"),
                git_commit=git, data_hash=panel.data_hash, n_assets=len(panel.tickers),
                first_date=str(panel.dates[0]), last_date=str(panel.dates[-1]),
                n_features=len(panel.fnames), argv=sys.argv, created=time.strftime("%Y-%m-%d %H:%M:%S"))


def main(cfg: Config):
    out = Path(cfg.out)
    for sub in ("scores", "tables", "figures", "diagnostics"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="a")])
    log.setLevel(logging.INFO)
    t_start = time.time()
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    apply_style()
    horizons = [int(x) for x in cfg.horizons.split(",")]
    models = [m.strip() for m in cfg.models.split(",")]
    grid = [float(x) for x in cfg.cost_grid.split(",")]
    if any(m in ("ran", "ran_nograph") for m in models) and torch is None and not cfg.check_data:
        raise SystemExit("PyTorch is required for the RAN models (pip install torch), or use --models ridge,mom,rev,random")

    # ---- data ----
    if cfg.synthetic > 0:
        log.warning("*** SYNTHETIC DATA - pipeline smoke test only, NOT valid for research results ***")
        prices, uni = make_synthetic(cfg.synthetic, cfg)
    else:
        uni = load_universe(cfg)
        prices = download_prices(list(uni["ticker"]), cfg)
    fields, sec, audit = build_master(prices, uni, cfg)
    audit.to_csv(out / "tables" / "universe_audit.csv", index=False)
    log.info("Universe: %d downloaded -> %d kept", len(prices), len(sec))
    if cfg.check_data:
        kept = audit[audit.status == "kept"]
        print(f"\nDATA CHECK OK: {len(kept)} tickers kept, {len(audit) - len(kept)} dropped")
        print(audit[audit.status != "kept"].to_string(index=False))
        print("sectors/asset classes:", sec["sector"].value_counts().to_dict())
        return
    panel = build_panel(fields, sec, cfg)
    if cfg.stage == "dev":
        panel = truncate_panel(panel, np.datetime64(cfg.holdout_start))
        log.info("STAGE=dev: data truncated before %s (holdout never loaded)", cfg.holdout_start)
    log.info("Panel: T=%d A=%d F=%d  %s -> %s", len(panel.dates), len(panel.tickers), len(panel.fnames),
             panel.dates[0], panel.dates[-1])
    manifest = env_manifest(cfg, panel)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    pd.DataFrame({"ticker": panel.tickers, "sector": panel.sectors}).to_csv(out / "tables" / "universe_final.csv", index=False)

    eng = None
    if any(m in ("ran", "ran_nograph") for m in models):
        dev = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else
                           (cfg.device if cfg.device != "auto" else "cpu"))
        log.info("Torch device: %s", dev)
        eng = TorchEngine(panel, cfg, dev)

    # ---- scores ----
    scores, Ys = {}, {}
    for h in horizons:
        y = make_target(panel.close, h, cfg.exec_lag)
        Ys[h] = y
        plan = plan_windows(panel, y, h, cfg)
        log.info("H=%d: %d walk-forward windows", h, len(plan))
        if not plan:
            raise SystemExit("No walk-forward windows: reduce --train_window/--min_train or extend the date range.")
        for m in models:
            scores[(m, h)] = get_scores(m, h, panel, y, plan, cfg, out, eng)
        # frozen-holdout ("one-shot") model, final stage only, learned models only
        if cfg.stage == "final":
            for m in [x for x in models if x in ("ran", "ran_nograph", "ridge")]:
                fp = fingerprint(cfg, panel.data_hash, h, m + "_frozen")
                path = out / "scores" / f"{m}_frozen_h{h}.npz"
                if cfg.resume and path.exists() and str(np.load(path, allow_pickle=True)["fp"]) == fp:
                    scores[(m + "_frozen", h)] = np.load(path, allow_pickle=True)["scores"]
                    continue
                S, dg = run_model(m, panel, y, frozen_plan(plan, panel, y, h, cfg), cfg, eng)
                np.savez_compressed(path, scores=S, fp=fp)
                scores[(m + "_frozen", h)] = S

    # ---- analysis ----
    periods = periods_for(cfg, panel)
    boot = BootCache(cfg.n_boot, cfg.seed)
    R, perm, cost_res = {}, {}, {}
    all_models = models + [k[0] for k in scores if k[0].endswith("_frozen") and k[1] == horizons[0]]
    all_models = list(dict.fromkeys(all_models))
    for label in all_models:
        MODEL_LABEL.setdefault(label, MODEL_LABEL.get(label.replace("_frozen", ""), label) + " (frozen)")
        MODEL_STYLE.setdefault(label, MODEL_STYLE.get(label.replace("_frozen", ""), MODEL_STYLE["ran"]))
    for h in horizons:
        for m in all_models:
            S = scores[(m, h)]
            for pname, pmask in periods.items():
                if m.endswith("_frozen") and pname != "holdout":
                    continue
                for var in ("raw", "sector"):
                    r = analyse(S, Ys[h], h, panel, cfg, pmask, var, boot)
                    if r is not None:
                        R[(pname, m, h, var)] = r
        for m in models:
            r = R.get(("full", m, h, "raw"))
            if r is None:
                continue
            r["_dec"] = decile_profile(r["_S"], Ys[h])
            cost_res[(m, h)] = cost_sensitivity(r["_S"], Ys[h], h, cfg, grid)
            if m != "random":
                perm[(m, h)] = permutation_test(r["_S"], Ys[h], h, cfg.top_frac, cfg.n_perm, cfg.seed)
        log.info("Analysis done for H=%d", h)

    # Holm over family of one-sided IC tests (raw variant), per period
    for pname in periods:
        fam = {(m, h): R[(pname, m, h, "raw")]["ic_p_nw"] for m in all_models if m != "random" for h in horizons
               if (pname, m, h, "raw") in R}
        adj = holm(fam)
        for (m, h), p in adj.items():
            R[(pname, m, h, "raw")]["ic_p_holm"] = p

    # ---- tables ----
    cols = ["horizon", "model", "variant", "ic", "ic_t_nw", "ic_p_nw", "ic_p_holm", "icir", "sharpe_gross",
            "sharpe_net", "sharpe_lo", "sharpe_hi", "psr", "ann_ret", "ann_vol", "turnover", "max_dd", "n_rows", "n_indep"]
    hdr = {"ic": "Rank IC", "ic_t_nw": "NW t", "ic_p_nw": "p (1-sided)", "ic_p_holm": "Holm p", "icir": "ICIR",
           "sharpe_gross": "Sharpe (gross)", "sharpe_net": "Sharpe (net)", "sharpe_lo": "CI low", "sharpe_hi": "CI high",
           "psr": "PSR", "ann_ret": "Ann. ret", "ann_vol": "Ann. vol", "turnover": "Turnover", "max_dd": "Max DD",
           "n_rows": "N rows", "n_indep": "N indep."}
    for pname in periods:
        rows = []
        for h in horizons:
            for m in all_models:
                for var in ("raw", "sector"):
                    r = R.get((pname, m, h, var))
                    if r is None:
                        continue
                    row = dict(horizon=h, model=MODEL_LABEL[m], variant="raw" if var == "raw" else "sector-neutral")
                    for c in cols[3:]:
                        row[c] = r.get(c, np.nan)
                    rows.append(row)
        df = pd.DataFrame(rows, columns=cols).rename(columns=hdr)
        save_table(df, out, f"main_results_{pname}",
                   f"Out-of-sample results ({pname}); net of {cfg.cost_bps:g} bps/side costs and {cfg.borrow_bps:g} bps borrow.")
        log.info("\n=== MAIN RESULTS [%s] ===\n%s", pname, df.round(3).to_string(index=False))

    # paired comparisons
    rows = []
    for h in horizons:
        for base in [x for x in models if x not in ("ran", "random")]:
            a, b = R.get(("full", "ran", h, "raw")), R.get(("full", base, h, "raw"))
            if a is None or b is None or len(a["_net"]) != len(b["_net"]):
                continue
            n, L = len(a["_net"]), max(2 * h, 5)
            idx = boot.get(n, L)
            d_sh = boot_sharpe(a["_net"], h, idx) - boot_sharpe(b["_net"], h, idx)
            d_ic = a["_ic"][idx].mean(1) - b["_ic"][idx].mean(1)
            rows.append(dict(horizon=h, comparison=f"RAN - {MODEL_LABEL[base]}",
                             d_ic=a["ic"] - b["ic"], d_ic_lo=np.percentile(d_ic, 2.5), d_ic_hi=np.percentile(d_ic, 97.5),
                             p_ic=(d_ic <= 0).mean(), d_sharpe=a["sharpe_net"] - b["sharpe_net"],
                             d_sh_lo=np.percentile(d_sh, 2.5), d_sh_hi=np.percentile(d_sh, 97.5), p_sh=(d_sh <= 0).mean()))
    if rows:
        df = pd.DataFrame(rows)
        save_table(df, out, "paired_comparisons",
                   "Paired stationary-bootstrap differences (RAN minus comparator); p = bootstrap P(diff <= 0).")
        log.info("\n=== PAIRED COMPARISONS ===\n%s", df.round(3).to_string(index=False))

    # cost sensitivity
    rows = [dict(horizon=h, model=MODEL_LABEL[m], **{f"{g:g} bps": cost_res[(m, h)][g] for g in grid})
            for h in horizons for m in models if (m, h) in cost_res]
    save_table(pd.DataFrame(rows), out, "cost_sensitivity", "Net Sharpe ratio versus one-way transaction cost.")

    # permutation
    rows = [dict(horizon=h, model=MODEL_LABEL[m], gross_sharpe=p["actual"], p_median=p["p_median"],
                 p_by_offset=", ".join(f"{x:.3f}" for x in p["p_all"])) for (m, h), p in perm.items()]
    save_table(pd.DataFrame(rows), out, "permutation_tests",
               "Within-day permutation test on non-overlapping rebalances (median p over start offsets).")

    # yearly
    rows = []
    for h in horizons:
        for m in [x for x in models if x in ("ran", "ridge", "ran_nograph")]:
            r = R.get(("full", m, h, "raw"))
            if r is None:
                continue
            yrs = pd.to_datetime(r["_dates"]).year
            for yv in np.unique(yrs):
                k = yrs == yv
                rows.append(dict(horizon=h, model=MODEL_LABEL[m], year=int(yv), rank_ic=r["_ic"][k].mean(),
                                 ann_net_ret=r["_net"][k].mean() * 252 / h, n_rows=int(k.sum())))
    save_table(pd.DataFrame(rows), out, "yearly_breakdown", "Calendar-year breakdown of rank IC and annualised net return.")

    # stress episodes
    rows = []
    for h in horizons:
        for m in [x for x in models if x in ("ran", "ridge", "mom")]:
            r = R.get(("full", m, h, "raw"))
            if r is None:
                continue
            d = pd.to_datetime(r["_dates"])
            for name, (a, b) in EPISODES.items():
                k = (d >= a) & (d <= b)
                if k.sum() >= 10:
                    rows.append(dict(horizon=h, model=MODEL_LABEL[m], episode=name, rank_ic=r["_ic"][k].mean(),
                                     ann_net_ret=r["_net"][k].mean() * 252 / h, n_rows=int(k.sum())))
    if rows:
        save_table(pd.DataFrame(rows), out, "stress_episodes", "Performance by market-stress episode (entry-date windows).")

    # benchmark + universe
    ew = (panel.close[1:] / panel.close[:-1] - 1).mean(1)
    info = dict(n_assets=len(panel.tickers), n_dates=len(panel.dates), first=str(panel.dates[0]), last=str(panel.dates[-1]),
                n_sectors=int(len(np.unique(panel.sectors))), ew_ann_ret=float(ew.mean() * 252),
                ew_ann_vol=float(ew.std() * np.sqrt(252)), ew_sharpe=float(ew.mean() / ew.std() * np.sqrt(252)),
                stage=cfg.stage, holdout_start=cfg.holdout_start)
    save_table(pd.DataFrame([info]), out, "universe_summary", "Universe and equal-weight benchmark summary.")

    # ---- figures ----
    main_models = [m for m in models]
    fig_equity(R, panel, cfg, main_models, horizons, out)
    fig_rolling_ic(R, cfg, main_models, horizons, out)
    for pname in periods:
        fig_sharpe_forest(R, [m for m in all_models], horizons, pname, out)
    fig_cost(cost_res, main_models, horizons, grid, out)
    fig_annual_ic(R, main_models, horizons, out)
    fig_perm(perm, horizons, out, "ran" if "ran" in models else models[0])
    fig_deciles(R, horizons, out, "ran" if "ran" in models else models[0])

    # ---- results.json (scalar metrics only) ----
    js = {"|".join(map(str, k)): {a: b for a, b in v.items() if not a.startswith("_")} for k, v in R.items()}
    (out / "results.json").write_text(json.dumps(js, indent=2, default=lambda o: float(o)))
    log.info("DONE in %.1f min. Artefacts in %s", (time.time() - t_start) / 60, out.resolve())


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        flag = f"--{f.name}"
        if f.type in ("bool", bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}[f.type if isinstance(f.type, str) else f.type.__name__]
            ap.add_argument(flag, type=typ, default=f.default)
    return Config(**vars(ap.parse_args()))


if __name__ == "__main__":
    main(parse_args())