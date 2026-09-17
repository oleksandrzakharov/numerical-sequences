"""
randomness_toolkit.py — diagnostics for predictable structure in a 1-D sequence of numbers.

Every diagnostic answers a DIFFERENT question and returns (statistic, p, B, kind):
  kind = "perm"     : Monte-Carlo permutation p-value with the +1 correction of
                      Phipson & Smyth (2010): p = (#{T_null >= T_real} + 1) / (B + 1).
                      The smallest reportable p is 1/(B+1). B is the permutation count.
  kind = "analytic" : asymptotic p-value from a stated limiting distribution; B is None.

Nulls (stated per test):
  * "iid" — the observations are exchangeable (the iid null implies this; exchangeability
    does not imply independence). Implemented by permuting the ORIGINAL sequence and
    recomputing the whole statistic, including lagged pairs, symbols and delay
    embeddings, from the permuted sequence. Rejection = evidence against exchangeability;
    it does not by itself say which kind of structure is present.
  * "stationary, weakly dependent, constant mean/scale" — for the CUSUM change tests,
    approximated analytically with a Bartlett long-run-variance estimate (Newey & West
    1987) and the sup|Brownian bridge| limit. Validity is checked by simulation in
    calibration_study.py; see the article's appendix for the limitations.

Notation: B = permutation count; K = number of test blocks; m = long-run-variance bandwidth.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------- helpers
def check_series(x, min_len: int = 30) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.ndim != 1 or len(x) < min_len:
        raise ValueError(f"need a 1-D sequence with at least {min_len} values, got {len(x)}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{np.sum(~np.isfinite(x))} non-finite values: handle missing data first")
    if np.ptp(x) == 0:
        raise ValueError("constant sequence: every diagnostic is undefined")
    return x


def perm_p(real: float, null) -> float:
    """Monte-Carlo p-value with +1 correction (Phipson & Smyth 2010). Floor = 1/(B+1).
    Non-finite draws are an error (they indicate a degenerate statistic), not something to drop."""
    null = np.asarray(null, dtype=float)
    if not np.isfinite(real):
        raise ValueError("observed statistic is not finite")
    bad = ~np.isfinite(null)
    if bad.any():
        raise ValueError(f"{bad.sum()} of {len(null)} null draws are non-finite; diagnose the statistic")
    return float((np.sum(null >= real) + 1) / (len(null) + 1))


def fmt_p(p: float, B) -> str:
    if B is None:                      # analytic p-value
        return "p<0.001a" if p < 0.0005 else f"p={p:.3f}a"
    floor = 1 / (B + 1)
    return f"p<={floor:.3f}" if p <= floor + 1e-12 else f"p={p:.3f}"


def _perm_seq(x: np.ndarray) -> np.ndarray:
    return RNG.permutation(x)


def _perm_within(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Permute values only among positions that share a label (e.g. hour of day). Null: given the
    label, the values are exchangeable. Keeps every calendar mean exactly; destroys serial
    dependence. Use when the question is 'predictability beyond the clock'."""
    out = np.array(x, dtype=float, copy=True)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        out[idx] = x[RNG.permutation(idx)]
    return out


# ----------------------------------------------------------------------------- 1. serial correlation
def autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    if not 1 <= lag < len(x):
        raise ValueError(f"lag must be in [1, n-1], got {lag}")
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def ljung_box_stat(x: np.ndarray, max_lag: int = 10) -> float:
    n = len(x)
    return float(n * (n + 2) * sum(autocorr(x, k) ** 2 / (n - k) for k in range(1, max_lag + 1)))


def ljung_box_analytic(x: np.ndarray, max_lag: int = 10, fitted_params: int = 0):
    """Ljung & Box (1978). Asymptotic chi-square reference with (max_lag - fitted_params) df
    when Q is computed on residuals of a fitted ARMA model. Assumes white noise with constant
    variance; the approximation is asymptotic and distorted under conditional
    heteroskedasticity. The workflow uses the permutation version instead."""
    q = ljung_box_stat(x, max_lag)
    df = max_lag - fitted_params
    if df <= 0:
        raise ValueError("max_lag must exceed the number of fitted parameters")
    return q, float(stats.chi2.sf(q, df)), None, "analytic"


def ljung_box_test(x: np.ndarray, max_lag: int = 10, B: int = 500):
    """Null: exchangeable (iid). Statistic: Q over lags 1..max_lag. Resampling: permute the sequence."""
    x = check_series(x)
    real = ljung_box_stat(x, max_lag)
    null = [ljung_box_stat(_perm_seq(x), max_lag) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


def max_acf_test(x: np.ndarray, max_lag: int = 20, B: int = 500):
    """Null: iid. Statistic: max_k |rho(k)|, k = 1..max_lag; the null repeats the same search."""
    x = check_series(x)

    def stat(v):
        return max(abs(autocorr(v, k)) for k in range(1, max_lag + 1))
    real = stat(x)
    null = [stat(_perm_seq(x)) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


def runs_stat(x: np.ndarray) -> float:
    """Wald–Wolfowitz (1940) runs z on above/below-median indicators. Values equal to the
    median count as 'below'. Returns |z| so that too few and too many runs both count."""
    s = x > np.median(x)
    n1, n2 = int(s.sum()), int((~s).sum())
    if n1 == 0 or n2 == 0:
        return 0.0
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return abs(float((runs - mu) / np.sqrt(var))) if var > 0 else 0.0


def runs_test(x: np.ndarray, B: int = 500):
    """Null: iid. Statistic: |z| of the runs count. Resampling: permute the sequence."""
    x = check_series(x)
    real = runs_stat(x)
    null = [runs_stat(_perm_seq(x)) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


# ----------------------------------------------------------------------------- 2. symbolic dependence
def discretize(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Equal-frequency bins -> symbols 0..k-1. Heavy ties make the bins unequal (and may
    empty a bin); the permutation null is unaffected because the symbol multiset is kept."""
    if len(np.unique(x)) < k:
        raise ValueError(f"fewer than {k} distinct values; cannot form {k} states")
    qs = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(qs, x, side="right")


def transition_g(sym: np.ndarray, k: int) -> float:
    """G-statistic of the one-step transition table against row/column independence."""
    m = np.zeros((k, k))
    np.add.at(m, (sym[:-1], sym[1:]), 1)
    expected = m.sum(1, keepdims=True) * m.sum(0, keepdims=True) / m.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(m > 0, m * np.log(m / expected), 0.0)
    return float(2 * terms.sum())


def transition_test(x: np.ndarray, k: int = 3, B: int = 500):
    """Null: iid symbols. Statistic: G. Resampling: permute the symbol sequence."""
    x = check_series(x)
    sym = discretize(x, k)
    real = transition_g(sym, k)
    null = [transition_g(_perm_seq(sym), k) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


def ngram_best_z(sym: np.ndarray, k: int, order: int, min_support: int = 20) -> float:
    """Mine every context of length `order`; return the largest binomial |z| over all
    (context -> next symbol) cells with support >= min_support. The z-score is a RANKING
    heuristic (context occurrences overlap, so the binomial model is not exact); the
    permutation null supplies the reference distribution."""
    n = len(sym)
    base = np.bincount(sym, minlength=k) / n
    ctx = np.zeros(n - order, dtype=np.int64)
    for j in range(order):
        ctx = ctx * k + sym[j:n - order + j]
    nxt = sym[order:]
    best = 0.0
    for c in np.unique(ctx):
        sel = nxt[ctx == c]
        m = len(sel)
        if m < min_support:
            continue
        counts = np.bincount(sel, minlength=k)
        for s in range(k):
            p = base[s]
            if p <= 0 or p >= 1:
                continue
            z = (counts[s] - m * p) / np.sqrt(m * p * (1 - p))
            best = max(best, abs(float(z)))
    return best


def ngram_search(sym: np.ndarray, k: int, max_order: int) -> float:
    return max(ngram_best_z(sym, k, o) for o in range(1, max_order + 1))


def ngram_test(x: np.ndarray, k: int = 3, max_order: int = 3, B: int = 200):
    """Null: iid symbols. Statistic: best |z| over ALL contexts of ALL orders (max-statistic).
    Resampling: ONE permuted sequence per replicate; the whole search is repeated on it."""
    x = check_series(x)
    sym = discretize(x, k)
    real = ngram_search(sym, k, max_order)
    null = [ngram_search(_perm_seq(sym), k, max_order) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


# ----------------------------------------------------------------------------- 3. general lag-1 dependence
def distance_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Sample distance correlation (Székely, Rizzo & Bakirov 2007). The POPULATION dCor is zero
    iff X and Y are independent, given finite first moments; the sample value is biased
    upward, and a finite-sample non-rejection does not establish independence."""
    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(0, keepdims=True) - d.mean(1, keepdims=True) + d.mean()
    A, B_ = centred(a), centred(b)
    dvar_a, dvar_b = (A * A).mean(), (B_ * B_).mean()
    if dvar_a <= 0 or dvar_b <= 0:
        raise ValueError("distance variance is zero (constant input)")
    return float(np.sqrt(max((A * B_).mean(), 0) / np.sqrt(dvar_a * dvar_b)))


def dcor_lag_test(x: np.ndarray, lag: int = 1, B: int = 300, max_n: int = 1500):
    """Null: iid sequence. Statistic: dCor(x_t, x_{t-lag}) on a fixed subsample of positions.
    Resampling: permute the ORIGINAL sequence and rebuild the lagged pairs on the same
    positions (permuting one column of already-built pairs ignores that pairs overlap)."""
    x = check_series(x)
    m = len(x) - lag
    if m < 10:
        raise ValueError("too few lagged pairs")
    idx = np.arange(m) if m <= max_n else np.sort(RNG.choice(m, max_n, replace=False))

    def stat(v):
        return distance_correlation(v[lag:][idx], v[:-lag][idx])
    real = stat(x)
    null = [stat(_perm_seq(x)) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


# ----------------------------------------------------------------------------- 4. out-of-sample forecasting
def embed(x: np.ndarray, lags: int, horizon: int = 1):
    """Row for target index t: (x_{t-h}, x_{t-h-1}, ..., x_{t-h-lags+1}). Returns X and the
    target positions t = lags + h - 1 ... n - 1."""
    n = len(x)
    start = lags + horizon - 1
    X = np.column_stack([x[start - horizon - j: n - horizon - j] for j in range(lags)])
    return X, np.arange(start, n)


def _knn_predict(Xtr, ytr, Xte, k=25):
    k = min(k, len(ytr))
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    return ytr[nn].mean(1)


def rolling_origin_eval(x: np.ndarray, lags: int = 3, target: str = "mean", horizon: int = 1,
                        n_folds: int = 5, first_train_frac: float = 0.4, k: int = 25,
                        extra_purge: int = 0, season: np.ndarray | None = None) -> dict:
    """Sequential h-step-ahead prediction with a model frozen at each fold's origin.

    Fold f has origin o_f. Training rows are targets with index t <= o_f - extra_purge
    (the outcome is fully observed at the origin; extra_purge > 0 is for window targets that
    end after t). Test rows are o_f < t <= o_{f+1}; their features (x_{t-h}, ...) are
    available when that forecast is made. No other gap is imposed: with past-only features
    and point targets, a training outcome may legitimately be a later forecast input.

    target='mean'      : y_t = x_t
    target='magnitude' : y_t = |x_t - c_f|, features |X - c_f|, where c_f is the mean of the
                         training window x[:o_f]  (learned at the origin, not globally).
    season             : optional integer labels (hour of day, weekday ...). At each origin the
                         per-label means are estimated from the training window x[:o_f] only and
                         subtracted from the whole series before embedding. Learned preprocessing
                         never sees the test fold.

    Returns skill = 1 - MSE(model)/MSE(mean baseline) for the model and for persistence
    (y_hat_t = y_{t-1}), pooled over folds and per fold, plus MSEs."""
    x_raw = np.asarray(x, float)
    n_all = len(x_raw)
    pos = np.arange(lags + horizon - 1, n_all)
    n = len(pos)
    edges = np.linspace(int(first_train_frac * n), n, n_folds + 1).astype(int)
    sse = {"model": 0.0, "mean": 0.0, "persist": 0.0}
    per_fold = []
    for f in range(n_folds):
        lo, hi = edges[f], edges[f + 1]
        origin = pos[lo] - 1                       # last observed index at the origin
        tr = np.where(pos <= origin - extra_purge)[0]
        te = np.arange(lo, hi)
        if len(te) == 0 or len(tr) < k:
            continue
        x = x_raw
        if season is not None:                     # calendar adjustment learned from the training window
            season = np.asarray(season)
            adj = np.zeros(n_all)
            for lab in np.unique(season):
                m = season[:origin + 1] == lab
                adj[season == lab] = x_raw[:origin + 1][m].mean() if m.any() else x_raw[:origin + 1].mean()
            x = x_raw - adj
        X, _ = embed(x, lags, horizon)
        if target == "mean":
            y = x[pos]; Xf = X
        elif target == "magnitude":
            c = x[:origin + 1].mean()
            y = np.abs(x[pos] - c); Xf = np.abs(X - c)
        else:
            raise ValueError(target)
        pred = _knn_predict(Xf[tr], y[tr], Xf[te], k)
        e_m = ((y[te] - pred) ** 2).sum()
        e_b = ((y[te] - y[tr].mean()) ** 2).sum()
        e_p = ((y[te] - y[te - horizon]) ** 2).sum()
        sse["model"] += e_m; sse["mean"] += e_b; sse["persist"] += e_p
        per_fold.append(1 - e_m / e_b if e_b > 0 else np.nan)
    if sse["mean"] <= 0:
        raise ValueError("baseline SSE is zero (constant target)")
    return {"skill_model": 1 - sse["model"] / sse["mean"],
            "skill_persistence": 1 - sse["persist"] / sse["mean"],
            "skill_per_fold": per_fold,
            "mse_model": sse["model"], "mse_mean": sse["mean"], "mse_persist": sse["persist"]}


def forecast_test(x: np.ndarray, target: str = "mean", lags: int = 3, B: int = 100,
                  null: str = "iid", **kw):
    """DETECTION test. Statistic: rolling-origin kNN skill vs the mean baseline. Resampling:
    generate a null sequence and rebuild the entire pipeline on it (embedding, calendar
    adjustment, centring, folds, fit, score).
      null="iid"           : permute the whole sequence. Tests exchangeability of the values;
                             if `season` is given, the null ALSO removes the calendar cycle, so a
                             rejection can come from the cycle alone.
      null="within_season" : permute only within calendar strata (requires `season`). Tests
                             'exchangeable given the clock' - the question of predictability
                             BEYOND the calendar means, which are kept exactly.
    A negative skill (does not beat the baseline) can still exceed the null pipelines: report
    detection and improvement separately."""
    x = check_series(x)
    if null == "within_season":
        if kw.get("season") is None:
            raise ValueError("null='within_season' needs season labels")
        labels = np.asarray(kw["season"])
        draw = lambda: _perm_within(x, labels)
    elif null == "iid":
        draw = _perm_seq_of(x)
    else:
        raise ValueError(null)
    real = rolling_origin_eval(x, lags, target, **kw)["skill_model"]
    null_draws = [rolling_origin_eval(draw(), lags, target, **kw)["skill_model"] for _ in range(B)]
    return real, perm_p(real, null_draws), B, "perm"


def _perm_seq_of(x):
    return lambda: _perm_seq(x)


# ----------------------------------------------------------------------------- 5. changes in level and scale
def block_mean_spread(x: np.ndarray, K: int) -> float:
    return float(np.std([b.mean() for b in np.array_split(x, K)]))


def block_mean_test(x: np.ndarray, K: int = 20, B: int = 500):
    """Null: exchangeable (iid). Statistic: sd of the K consecutive block means.
    Rejection = not exchangeable: EITHER a changing level OR ordinary serial dependence
    (a stationary AR(1) inflates the spread of block means). It does not separate the two."""
    x = check_series(x)
    real = block_mean_spread(x, K)
    null = [block_mean_spread(_perm_seq(x), K) for _ in range(B)]
    return real, perm_p(real, null), B, "perm"


def bartlett_lrv(v: np.ndarray, m: int) -> float:
    """Bartlett-kernel long-run variance (Newey & West 1987) with bandwidth m lags."""
    v = v - v.mean(); n = len(v)
    s = v @ v / n
    for k in range(1, m + 1):
        s += 2 * (1 - k / (m + 1)) * (v[:-k] @ v[k:] / n)
    return float(s)


def andrews_bandwidth(v: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel:
    m = 1.1447 (alpha n)^(1/3), alpha = 4 rho^2 / ((1-rho)^2 (1+rho)^2)."""
    rho = autocorr(v, 1)
    rho = float(np.clip(rho, -0.97, 0.97))
    alpha = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return max(1, int(np.floor(1.1447 * (alpha * len(v)) ** (1 / 3))))


def cusum_stat(v: np.ndarray, m: int) -> float:
    """T = max_k |S_k| / (sigma_LR sqrt(n)), S_k = cumulative sum of (v_t - mean).
    Under a stationary, weakly dependent, constant-mean null, T -> sup|Brownian bridge|
    (Ploberger & Krämer 1992 for the OLS-residual CUSUM)."""
    n = len(v)
    lrv = bartlett_lrv(v, m)
    if lrv <= 0:
        raise ValueError("long-run variance estimate is not positive")
    S = np.cumsum(v - v.mean())
    return float(np.max(np.abs(S)) / np.sqrt(lrv * n))


def cusum_mean_test(x: np.ndarray, m: int | None = None):
    """Change-in-LEVEL diagnostic. Null: stationary, weakly dependent, constant mean.
    Analytic p from the Kolmogorov distribution (sup|Brownian bridge|). m = bandwidth of
    the long-run variance (default: Andrews AR(1) plug-in). Approximate: see calibration."""
    x = check_series(x)
    m = andrews_bandwidth(x) if m is None else m
    T = cusum_stat(x, m)
    return T, float(stats.kstwobign.sf(T)), None, "analytic"


def cusum_scale_test(x: np.ndarray, m: int | None = None):
    """Change-in-SCALE diagnostic: the same CUSUM applied to squared deviations from the mean
    (Inclán & Tiao 1994, with a long-run-variance correction for dependence in the squares).
    Bandwidth default m = floor(sqrt(n)): squared deviations of a variance-clustering process are
    persistently correlated and the AR(1) plug-in under-covers them (rejecting a stationary
    clustered process ~50% of the time); with sqrt(n) the simulated size is 8-9% on that process
    and 3-6% on iid / AR(1) (see scale_test_options.py). EXPLORATORY under persistent variance
    clustering: the 8-11% size is about twice nominal, so a borderline p there is not evidence
    of a variance change (follow up with the self-normalised statistic). A change in level also moves this
    statistic; run the level test first."""
    x = check_series(x)
    v = (x - x.mean()) ** 2
    m = int(np.sqrt(len(v))) if m is None else m
    T = cusum_stat(v, m)
    return T, float(stats.kstwobign.sf(T)), None, "analytic"


# ----------------------------------------------------------------------------- the workflow (exploratory battery)
def workflow(x: np.ndarray, B_cheap: int = 500, B_mid: int = 300, B_dear: int = 100) -> dict:
    """Runs every diagnostic. This is the EXPLORATORY battery used for the figures; the article
    recommends choosing a small subset for a given decision."""
    x = check_series(x)
    return {
        "ljung_box_10":       ljung_box_test(x, 10, B_cheap),
        "max_acf_20":         max_acf_test(x, 20, B_cheap),
        "runs":               runs_test(x, B_cheap),
        "transition_3state":  transition_test(x, 3, B_cheap),
        "ngram_max_z":        ngram_test(x, 3, 3, B_dear * 2),
        "dcor_lag1":          dcor_lag_test(x, 1, B_mid),
        "forecast_mean":      forecast_test(x, "mean", 3, B_dear),
        "forecast_magnitude": forecast_test(x, "magnitude", 3, B_dear),
        "block_mean_iid":     block_mean_test(x, 20, B_cheap),
        "cusum_mean":         cusum_mean_test(x),
        "cusum_scale":        cusum_scale_test(x),
    }


# ----------------------------------------------------------------------------- synthetic sequences
def make_sequences(n: int = 2000, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """All sequences are SIMULATED. Nothing here comes from a real dataset."""
    r = RNG if rng is None else rng
    seqs = {}
    seqs["iid noise"] = r.standard_normal(n)

    for phi in (0.3, 0.7):
        ar = np.zeros(n); e = r.standard_normal(n)
        for t in range(1, n):
            ar[t] = phi * ar[t - 1] + e[t]
        seqs[f"AR(1) phi={phi}"] = ar

    lm = np.zeros(n); lm[0] = 0.37
    for t in range(1, n):
        lm[t] = 3.99 * lm[t - 1] * (1 - lm[t - 1])
    seqs["logistic map (chaos)"] = lm

    quad = np.zeros(n); e = r.standard_normal(n)
    for t in range(1, n):
        quad[t] = 0.6 * (min(quad[t - 1] ** 2, 4.0) - 1.0) + e[t]
    seqs["quadratic memory"] = quad

    vc = np.zeros(n); sig2 = 1.0; e = r.standard_normal(n)
    for t in range(n):
        sig2 = 0.05 + 0.10 * (vc[t - 1] ** 2 if t else 0) + 0.85 * sig2
        vc[t] = np.sqrt(sig2) * e[t]
    seqs["variance clustering"] = vc

    seqs["noise + strong regimes"] = r.standard_normal(n) + np.repeat(r.normal(0, 0.5, 20), n // 20)
    seqs["noise + weak regimes"] = r.standard_normal(n) + np.repeat(np.array([-0.2, 0.1, -0.1, 0.2]), n // 4)
    return seqs


if __name__ == "__main__":
    import json, time
    t0 = time.time()
    seqs = make_sequences()
    names = None
    rows = {}
    for label, x in seqs.items():
        res = workflow(x)
        if names is None:
            names = list(res.keys())
            print("sequence".ljust(24) + "".join(n[:17].ljust(19) for n in names))
        rows[label] = {k: [float(s), float(p), (None if b is None else int(b)), kind] for k, (s, p, b, kind) in res.items()}
        print(label.ljust(24) + "".join(f"{s:7.3f} {fmt_p(p, b)}".ljust(19) for s, p, b, _ in res.values()))
    json.dump(rows, open("toolkit_results.json", "w"), indent=1)
    np.savez("toolkit_sequences.npz", **seqs)
    print(f"\n{time.time() - t0:.0f}s   (a = analytic p-value; others are permutation p-values with floor 1/(B+1))")
