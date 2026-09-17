"""
sequence_report.py — one CSV in, one PDF report out, in the structure of the article's example report.
Works for any evenly sampled numerical sequence and any threshold-type event.

    python3 sequence_report.py data.csv --value load --time ts \\
        --threshold 0.86 --direction above --existing 0.80 --cost-act 1 --cost-miss 20 --out report.pdf

Pipeline (the article's order):
  1. data checks: delimiter and decimal detection, timestamps monotone, sampling interval, gaps, missing values
  2. optional transform (--transform diff | pct | logdiff) for level-like series; the report warns when
     a level-like series is analysed untransformed (persistence skill above --level-warn)
  3. reserve the last --reserve fraction, untouched until the end
  4. known structure: a calendar cycle inferred from the timestamps (time-of-day slots for sub-daily
     data, day-of-week for daily; --period N overrides, --period 0 disables), always estimated inside the
     training window; baselines (persistence, seasonal persistence); rolling-origin linear AR(--lags)
     at horizon --horizon with per-fold skill (--folds folds after a --first-train initial window)
  5. diagnostics with their nulls: Ljung–Box (permutation), rolling-origin kNN forecast skill (null =
     exchangeable given the clock when a cycle exists, else iid), CUSUM level, CUSUM scale (exploratory)
  6. with --threshold, --existing and --cost-miss: predicted event probability --horizon steps ahead
     (Gaussian residual), out-of-fold reliability in bins built around the break-even probability, with
     cluster-robust intervals (clusters = calendar days, or --cluster rows), and four policies on the
     reserved period — A never act, B the existing rule, B′ the existing rule re-tuned on development,
     C act when P(event) > cost_act / cost_miss — with paired cluster intervals for C−B, B′−B, C−B′
  6a. scaling back down: a two-level rule H (trigger level up, lower release level down) re-tuned on the
     development data with the switching cost --cost-switch (0 = changes are free), compared with B′
  6b. event risk by time of the cycle: the development event rate per slot (smoothed), the windows where
     it exceeds the break-even chance, and a fixed schedule S that acts only there, evaluated with the rest
  7. a reading that follows stated rules (printed in the report): a pattern counts as usable only if
     skill ≥ --min-skill in every fold; a candidate beats a policy only if its paired interval excludes
     zero; between candidates the data cannot separate, the simpler is preferred as a practical judgment

Event: --direction above (value > threshold), below (value < threshold) or abs (|value| > threshold).
The existing rule B acts when the current value is above / below / beyond --existing in the same sense.
CSV: header row; delimiter and decimal comma auto-detected; --value names the numeric column (default:
first numeric column that is not the time column); --time names an ISO-8601 timestamp column (optional;
without it rows are evenly spaced and clusters are blocks of --cluster rows).
Requires numpy, scipy, matplotlib and randomness_toolkit.py next to this file. PDF via headless
Chrome/Chromium if found; otherwise the HTML is written and said so.
"""
import argparse, base64, csv, io, json, os, shutil, subprocess, sys, time, pathlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import randomness_toolkit as rt


# ----------------------------------------------------------------------------- input
def _to_float(v):
    v = v.strip()
    if v == "" or v.lower() in ("na", "nan", "null", "none", "-"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return float(v.replace(" ", "").replace(",", "."))   # decimal comma


def read_csv(path, value_col, time_col, sep):
    raw = open(path, newline="", encoding="utf-8-sig").read()
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    rows = [r for r in csv.reader(io.StringIO(raw), delimiter=sep) if r]
    header = [h.strip() for h in rows[0]]; body = rows[1:]
    cols = {h: [r[i] if i < len(r) else "" for r in body] for i, h in enumerate(header)}
    if time_col is None:
        import re as _re
        for h in header:
            head50 = [s.strip() for s in cols[h][:50]]
            if not head50 or not all(_re.match(r"^\d{4}-\d{2}-\d{2}", s) for s in head50):
                continue
            try:
                np.array([s.replace(" ", "T") for s in head50], dtype="datetime64[s]"); time_col = h; break
            except Exception:
                pass
    t = None
    if time_col is not None:
        try:
            t = np.array([s.strip().replace(" ", "T") for s in cols[time_col]], dtype="datetime64[s]")
        except Exception as e:
            sys.exit(f"could not parse --time column {time_col!r} as ISO-8601 timestamps: {e}")
    if value_col is None:
        for h in header:
            if h == time_col:
                continue
            try:
                np.array([_to_float(v) for v in cols[h]], dtype=float); value_col = h; break
            except Exception:
                pass
        if value_col is None:
            sys.exit("no numeric column found; pass --value")
    try:
        x = np.array([_to_float(v) for v in cols[value_col]], dtype=float)
    except Exception as e:
        sys.exit(f"column {value_col!r} is not numeric: {e}")
    return x, t, value_col, time_col, sep


def data_checks(x, t, period_arg):
    notes = []
    n = len(x)
    n_missing = int(np.isnan(x).sum())
    if n_missing:
        notes.append(f"{n_missing} missing values filled by linear interpolation (the tests treat filled values as observed)")
        idx = np.arange(n); m = ~np.isnan(x)
        x = np.interp(idx, idx[m], x[m])
    interval_s = None; gaps = 0; slots_per_day = None; period = None; monotone = True
    if t is not None:
        d = np.diff(t).astype("timedelta64[s]").astype(int)
        monotone = bool((d > 0).all())
        if not monotone:
            notes.append("timestamps are not strictly increasing — the row order was kept as given")
        if len(d):
            vals, counts = np.unique(d, return_counts=True)
            interval_s = int(vals[np.argmax(counts)])
            gaps = int((d != interval_s).sum())
            if gaps:
                notes.append(f"{gaps} steps differ from the modal interval ({interval_s} s); they are treated as consecutive")
            if interval_s > 0 and 86400 % interval_s == 0 and interval_s < 86400:
                slots_per_day = 86400 // interval_s
            elif interval_s == 86400:
                slots_per_day = 1
    if period_arg is not None:
        period = None if period_arg == 0 else period_arg
    elif slots_per_day and slots_per_day > 1:
        period = slots_per_day
    elif slots_per_day == 1:
        period = 7
    return x, dict(n=n, missing=n_missing, monotone=monotone, interval_s=interval_s, gaps=gaps,
                   slots_per_day=slots_per_day, period=period, notes=notes)


def season_labels(n, t, info):
    period = info["period"]
    if period is None:
        return None, "none"
    if t is not None and info["slots_per_day"] and period == info["slots_per_day"] and period > 1:
        secs = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(int)
        return (secs // info["interval_s"]).astype(int), f"time-of-day, {period} slots"
    if t is not None and info["slots_per_day"] == 1 and period == 7:
        return ((t.astype("datetime64[D]").astype(int) + 3) % 7).astype(int), "day-of-week"
    return (np.arange(n) % period).astype(int), f"position modulo {period}"


def cluster_labels(n, t, info, cluster_arg):
    if cluster_arg:
        return np.arange(n) // cluster_arg, f"blocks of {cluster_arg} rows"
    if t is not None:
        d = t.astype("datetime64[D]").astype(int)
        return d - d.min(), "calendar days"
    block = info["period"] or 24
    return np.arange(n) // block, f"blocks of {block} rows"


# ----------------------------------------------------------------------------- modelling pieces
def season_means(v, lab, upto, n_lab):
    m = np.full(n_lab, v[:upto].mean())
    for k in range(n_lab):
        sel = lab[:upto] == k
        if sel.any():
            m[k] = v[:upto][sel].mean()
    return m


def rolling_linear_ar(v, lab, n_lab, lags, h, period, n_folds, first_train_frac):
    m = len(v) - lags - h + 1
    edges = np.linspace(int(first_train_frac * m), m, n_folds + 1).astype(int)
    sse_m = sse_b = sse_p = sse_s = 0.0; per_fold = []; n_s = 0
    for f in range(n_folds):
        lo, hi = edges[f], edges[f + 1]
        origin = lo + lags + h - 2                        # last index whose target is in the training rows
        sm = season_means(v, lab, origin + 1, n_lab) if lab is not None else None
        r = v - sm[lab] if lab is not None else v - v[:origin + 1].mean()
        X, pos = rt.embed(r, lags, h); y = r[pos]
        tr, te = np.arange(0, lo), np.arange(lo, hi)
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred = np.column_stack([np.ones(len(te)), X[te]]) @ beta
        em = ((y[te] - pred) ** 2).sum(); eb = ((y[te] - y[tr].mean()) ** 2).sum()
        ep = ((y[te] - r[pos[te] - h]) ** 2).sum()          # persistence: last value known at decision time
        if period and period >= h:
            es = ((v[pos[te]] - v[pos[te] - period]) ** 2).sum(); n_s += 1
        else:
            es = 0.0
        sse_m += em; sse_b += eb; sse_p += ep; sse_s += es; per_fold.append(1 - em / eb)
    return dict(skill=1 - sse_m / sse_b, persistence=1 - sse_p / sse_b,
                seasonal_persistence=(1 - sse_s / sse_b) if n_s else None, per_fold=per_fold)


def gap_interval(sel, p, y, cl):
    d = cl[sel]; cls = np.unique(d)
    g = np.array([(y[sel][d == dd].astype(float) - p[sel][d == dd]).sum() for dd in cls])
    D = len(cls)
    if D < 3:
        return (y[sel].mean() - p[sel].mean(), np.nan, np.nan)
    tot = g.sum(); se = np.sqrt(D * g.var(ddof=1)); tq = stats.t.ppf(0.975, D - 1)
    return tot / len(sel), (tot - tq * se) / len(sel), (tot + tq * se) / len(sel)


def paired_clusters(cost_fn, a, b, ex, cl, sa=None, sb=None):
    dd = np.unique(cl)
    da = np.array([cost_fn(a[cl == d], ex[cl == d], None if sa is None else sa[cl == d]) for d in dd])
    db = np.array([cost_fn(b[cl == d], ex[cl == d], None if sb is None else sb[cl == d]) for d in dd])
    diff = da - db; D = len(diff)
    if D < 3:
        return dict(mean=diff.mean(), lo=np.nan, hi=np.nan, rho1=np.nan, nw=(np.nan, np.nan), D=D)
    se = diff.std(ddof=1) / np.sqrt(D); tq = stats.t.ppf(0.975, D - 1)
    se_hac = np.sqrt(rt.bartlett_lrv(diff, 2) / D)
    return dict(mean=diff.mean(), lo=diff.mean() - tq * se, hi=diff.mean() + tq * se, rho1=rt.autocorr(diff, 1),
                nw=(diff.mean() - 1.96 * se_hac, diff.mean() + 1.96 * se_hac), D=D)


# ----------------------------------------------------------------------------- helpers
def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def isnan(v): return v is None or (isinstance(v, float) and np.isnan(v))
def f3(v): return "—" if isnan(v) else f"{v:+.3f}"
def pct(v): return "—" if isnan(v) else f"{100 * v:.1f}%"
def pts(v): return "—" if isnan(v) else f"{100 * v:+.1f}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--value"); ap.add_argument("--time"); ap.add_argument("--sep", default=None, help="CSV delimiter (auto-detected if omitted)")
    ap.add_argument("--transform", choices=["none", "diff", "pct", "logdiff"], default="none", help="analyse changes instead of levels")
    ap.add_argument("--title", default=None); ap.add_argument("--units", default="", help="unit label for the value")
    ap.add_argument("--reserve", type=float, default=0.3, help="fraction of the data reserved for the final evaluation")
    ap.add_argument("--lags", type=int, default=3); ap.add_argument("--horizon", type=int, default=1, help="steps ahead the decision is made for")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--first-train", type=float, default=0.4); ap.add_argument("--k", type=int, default=25, help="kNN neighbours")
    ap.add_argument("--period", type=int, default=None, help="cycle length in rows (0 = none); inferred from timestamps if omitted")
    ap.add_argument("--cluster", type=int, default=None, help="rows per cluster for the paired intervals (default: calendar days, else the cycle length)")
    ap.add_argument("--threshold", type=float, default=None, help="event threshold")
    ap.add_argument("--direction", choices=["above", "below", "abs"], default="above", help="event: value above / below / |value| above the threshold")
    ap.add_argument("--existing", type=float, default=None, help="existing rule: act when the current value is above/below/beyond this (same sense as --direction)")
    ap.add_argument("--cost-act", "--cost-check", dest="cost_act", type=float, default=1.0, help="cost of acting for one step")
    ap.add_argument("--cost-miss", type=float, default=None, help="cost of an unhandled event in one step")
    ap.add_argument("--cost-switch", type=float, default=0.0, help="cost of each change of state (scaling up or back down); 0 = changes are free")
    ap.add_argument("--min-skill", type=float, default=0.01, help="skill below this counts as 'not usable'")
    ap.add_argument("--level-warn", type=float, default=0.9, help="persistence skill above this triggers the level-series warning")
    ap.add_argument("--B-lb", type=int, default=500); ap.add_argument("--B-forecast", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default="report.pdf")
    a = ap.parse_args()
    t0 = time.time()
    rt.RNG = np.random.default_rng(a.seed)
    h = a.horizon

    x_raw, t, vcol, tcol, sep = read_csv(a.csv, a.value, a.time, a.sep)
    x, info = data_checks(x_raw, t, a.period)
    if a.transform != "none":
        if a.transform == "diff":
            x = np.diff(x)
        elif a.transform == "pct":
            x = np.diff(x) / x[:-1]
        else:
            if (x <= 0).any():
                sys.exit("logdiff needs strictly positive values")
            x = np.diff(np.log(x))
        if t is not None:
            t = t[1:]
        info["n"] = len(x)
    n = len(x)
    if n < 200:
        sys.exit(f"only {n} rows; the workflow needs a few hundred at least")
    lab, season_desc = season_labels(n, t, info)
    n_lab = int(lab.max()) + 1 if lab is not None else 0
    cl, cluster_desc = cluster_labels(n, t, info, a.cluster)
    period = info["period"]
    unit = a.units or ({"none": vcol, "diff": f"change in {vcol}", "pct": f"% change in {vcol}", "logdiff": f"log change in {vcol}"}[a.transform])
    title = a.title or f"Sequence report: {unit}"
    tr_desc = {"none": "none (levels analysed as given)", "diff": "differences x[t] − x[t−1]", "pct": "percentage changes", "logdiff": "log changes ln(x[t]/x[t−1])"}[a.transform]

    # ---- reserve
    split = int((1 - a.reserve) * n)
    dev, conf = np.arange(split), np.arange(split, n)
    cl_dev, cl_conf = len(np.unique(cl[dev])), len(np.unique(cl[conf]))
    span = lambda idx: (str(t[idx[0]])[:16].replace("T", " "), str(t[idx[-1]])[:16].replace("T", " ")) if t is not None else (f"row {idx[0]}", f"row {idx[-1]}")

    # ---- known structure and baselines
    ev = rolling_linear_ar(x[dev], lab[dev] if lab is not None else None, n_lab, a.lags, h, period, a.folds, a.first_train)
    sm_dev = season_means(x, lab, split, n_lab) if lab is not None else None
    resid_dev = x[dev] - (sm_dev[lab[dev]] if lab is not None else x[dev].mean())
    levelish = ev["persistence"] > a.level_warn and a.transform == "none"
    stable = all(s_ > 0 for s_ in ev["per_fold"])

    # ---- diagnostics
    lb = rt.ljung_box_test(resid_dev, 10, B=a.B_lb)
    kw = dict(horizon=h, n_folds=a.folds, first_train_frac=a.first_train, k=a.k)
    if lab is not None:
        fc = rt.forecast_test(x[dev], "mean", a.lags, B=a.B_forecast, null="within_season", season=lab[dev], **kw)
        fc_null = "exchangeable given the clock (permutation within calendar strata; pipeline rebuilt)"
    else:
        fc = rt.forecast_test(x[dev], "mean", a.lags, B=a.B_forecast, null="iid", **kw)
        fc_null = "iid sequence (whole-sequence permutation; pipeline rebuilt)"
    cm = rt.cusum_mean_test(resid_dev)
    cs = rt.cusum_scale_test(resid_dev)
    knn_read = (("the past carries information about the value %d step%s ahead, and the forecaster beats the baseline" % (h, "" if h == 1 else "s")) if fc[0] > 0 else
                "detection without improvement: above the null pipelines, but the forecaster does not beat the baseline (negative skill)") if fc[1] < 0.05 \
        else "no forward predictability detected with this model, these lags, this horizon, this n"
    diagnostics = [
        ("Ljung–Box Q(10), deseasonalised", f"Q = {lb[0]:.1f}", rt.fmt_p(lb[1], lb[2]), f"exchangeability (permutation, B = {lb[2]})",
         "not exchangeable; low-order autocorrelation" if lb[1] < 0.05 else "no detectable low-order autocorrelation at this n"),
        (f"Rolling-origin kNN forecast skill, horizon {h}", f"skill = {fc[0]:+.3f}", rt.fmt_p(fc[1], fc[2]), fc_null + f", B = {fc[2]}", knn_read),
        ("CUSUM level (LRV-scaled)", f"T = {cm[0]:.2f}", f"p = {cm[1]:.3f} (analytic)", "stationary, weakly dependent, constant mean",
         "level change beyond weak dependence" if cm[1] < 0.05 else "no level shift detected (does not establish stationarity)"),
        ("CUSUM scale (√n bandwidth) — exploratory", f"T = {cs[0]:.2f}", f"p = {cs[1]:.3f} (analytic)", "same, on squared deviations; size ≈ 2× nominal under persistent clustering",
         "possible scale change — exploratory; confirm with the self-normalised statistic" if cs[1] < 0.05 else "no scale change detected"),
    ]
    pattern = lb[1] < 0.05 or fc[1] < 0.05
    usable = pattern and ev["skill"] >= a.min_skill and stable and not levelish

    # ---- decision block (optional)
    decision = None
    if a.threshold is not None and a.existing is not None and a.cost_miss is not None:
        TAU, C_ACT, C_MISS = a.threshold, a.cost_act, a.cost_miss
        P_STAR = C_ACT / C_MISS
        if a.direction == "above":
            event = x > TAU; ev_desc = f"value above {TAU:g}"; rule = lambda s_, th: s_ > th
            p_of = lambda f_, s_: 1 - stats.norm.cdf((TAU - f_) / s_)
        elif a.direction == "below":
            event = x < TAU; ev_desc = f"value below {TAU:g}"; rule = lambda s_, th: s_ < th
            p_of = lambda f_, s_: stats.norm.cdf((TAU - f_) / s_)
        else:
            event = np.abs(x) > TAU; ev_desc = f"|value| above {TAU:g}"; rule = lambda s_, th: np.abs(s_) > th
            p_of = lambda f_, s_: 1 - stats.norm.cdf((TAU - f_) / s_) + stats.norm.cdf((-TAU - f_) / s_)
        X_dev, pos_dev = rt.embed(resid_dev, a.lags, h)
        A = np.column_stack([np.ones(len(pos_dev)), X_dev])
        beta = np.linalg.lstsq(A, resid_dev[pos_dev], rcond=None)[0]
        sigma = np.std(resid_dev[pos_dev] - A @ beta, ddof=a.lags + 1)
        base_all = sm_dev[lab] if lab is not None else np.full(n, x[dev].mean())
        resid_all = x - base_all
        X_all, pos_all = rt.embed(resid_all, a.lags, h)
        fcast = np.full(n, np.nan)
        fcast[pos_all] = base_all[pos_all] + np.column_stack([np.ones(len(pos_all)), X_all]) @ beta   # forecast of x_t made at t-h
        p_ev = p_of(fcast, sigma)
        # out-of-fold probabilities on development
        p_oof = np.full(n, np.nan)
        m_dev = len(dev) - a.lags - h + 1
        edges = np.linspace(int(a.first_train * m_dev), m_dev, a.folds + 1).astype(int)
        for f in range(a.folds):
            lo, hi = edges[f], edges[f + 1]
            origin = lo + a.lags + h - 2
            b_f = season_means(x, lab, origin + 1, n_lab)[lab] if lab is not None else np.full(n, x[:origin + 1].mean())
            r_f = x - b_f
            Xf, posf = rt.embed(r_f, a.lags, h)
            tr = posf[posf <= origin]; te = posf[(posf > origin) & (posf <= hi + a.lags + h - 2)]
            Af = np.column_stack([np.ones(len(tr)), Xf[:len(tr)]])
            bf = np.linalg.lstsq(Af, r_f[tr], rcond=None)[0]
            sf = np.std(r_f[tr] - Af @ bf, ddof=a.lags + 1)
            fte = b_f[te] + np.column_stack([np.ones(len(te)), Xf[te - posf[0]]]) @ bf
            p_oof[te] = p_of(fte, sf)
        bins = sorted(set([0.0] + [min(1.0, P_STAR * m_) for m_ in (0.5, 1, 2, 4)] + [1.0]))
        def reliability(idx, p):
            rows = []
            for lo_, hi_ in zip(bins[:-1], bins[1:]):
                sel = idx[(p[idx] >= lo_) & (p[idx] < hi_)] if hi_ < 1 else idx[(p[idx] >= lo_) & (p[idx] <= hi_)]
                if len(sel):
                    k = int(event[sel].sum()); g = gap_interval(sel, p, event, cl)
                    rows.append(dict(lo=lo_, hi=hi_, n=len(sel), pred=float(p[sel].mean()), obs=k / len(sel), gap=g[0], glo=g[1], ghi=g[2]))
            return rows
        oof_idx = np.where(~np.isnan(p_oof))[0]
        rel_oof = reliability(oof_idx, p_oof)
        idx_dev = dev[a.lags + h - 1:]
        C_SW = a.cost_switch
        def sw_of(act): return np.r_[act[:1], act[1:] != act[:-1]]          # a change of state at position t (start: off)
        def cost(act, e, sw=None):
            if sw is None: sw = sw_of(act)
            return C_ACT * act.mean() + C_MISS * (e & ~act).mean() + C_SW * sw.mean()
        sig = np.r_[np.full(h, np.nan), x[:-h]]                    # value known at decision time (h steps before the target)
        sig_dev = sig[idx_dev]
        cand = np.unique(np.quantile(np.abs(sig_dev) if a.direction == "abs" else sig_dev, np.linspace(0, 1, 301)))
        costs_grid = [cost(rule(sig_dev, th), event[idx_dev]) for th in cand]
        thB_star = float(cand[int(np.argmin(costs_grid))])
        dev_costs = dict(A=cost(np.zeros(len(idx_dev), bool), event[idx_dev]), B=cost(rule(sig_dev, a.existing), event[idx_dev]),
                         Bp=min(costs_grid), C=cost(p_ev[idx_dev] > P_STAR, event[idx_dev]))
        sel = conf[conf >= a.lags + h - 1]
        pol = {"A": np.zeros(len(sel), bool), "B": rule(sig[sel], a.existing), "B′": rule(sig[sel], thB_star), "C": p_ev[sel] > P_STAR}
        # ---- two-level rule H: scale up when the signal passes the upper level, back down only when it falls below the lower one
        gfun = (lambda s_: s_) if a.direction == "above" else (lambda s_: -s_) if a.direction == "below" else np.abs
        def hyst(g_, tu_, td_):
            up_ = g_ > tu_; dn_ = g_ < td_
            last = np.maximum.accumulate(np.where(up_ | dn_, np.arange(len(g_)), -1))
            return np.where(last >= 0, up_[np.maximum(last, 0)], False)
        g_dev = gfun(sig_dev); g_cand = np.unique(np.r_[np.quantile(g_dev, np.linspace(0, 1, 61)), gfun(np.array([thB_star]))])
        best = (np.inf, None, None)
        for tu_ in g_cand:
            for td_ in g_cand[g_cand <= tu_][::-1]:                 # gap ascending: ties keep the narrowest gap (no release level unless it pays)
                c_ = cost(hyst(g_dev, tu_, td_), event[idx_dev])
                if c_ < best[0] - 1e-12: best = (c_, float(tu_), float(td_))
        H_up, H_dn = best[1], best[2]
        disp = (lambda v_: v_) if a.direction != "below" else (lambda v_: -v_)
        H_desc = (f"scale up when the value is {'above' if a.direction == 'above' else 'below' if a.direction == 'below' else 'beyond ±'} {disp(H_up):.4g}, "
                  f"scale back down when it is {'below' if a.direction == 'above' else 'above' if a.direction == 'below' else 'within ±'} {disp(H_dn):.4g}")
        pol["H"] = hyst(gfun(sig[sel]), H_up, H_dn)
        dev_costs["H"] = best[0]
        H_same = abs(H_up - H_dn) < 1e-12
        # ---- event risk by time of the cycle, and a fixed schedule S built from it (development only)
        sched = None
        if lab is not None and n_lab > 1:
            cnt_d = np.bincount(lab[idx_dev], minlength=n_lab).astype(float)
            rate_raw = np.bincount(lab[idx_dev], weights=event[idx_dev].astype(float), minlength=n_lab) / np.maximum(cnt_d, 1)
            w = max(1, int(round(n_lab / 20)));  w += (w % 2 == 0)
            off = np.arange(-(w // 2), w // 2 + 1)
            rate_sm = np.array([rate_raw[(k_ + off) % n_lab].mean() for k_ in range(n_lab)])
            on = rate_sm > P_STAR
            cnt_c = np.bincount(lab[sel], minlength=n_lab).astype(float)
            rate_conf = np.bincount(lab[sel], weights=event[sel].astype(float), minlength=n_lab) / np.maximum(cnt_c, 1)
            def runs(b):
                if b.all(): return [(0, len(b) - 1)]
                if not b.any(): return []
                start = int(np.argmin(b)); out = []; cur = None
                for i_ in (np.arange(len(b)) + start) % len(b):
                    if b[i_]: cur = [i_, i_] if cur is None else [cur[0], i_]
                    elif cur is not None: out.append(tuple(cur)); cur = None
                if cur is not None: out.append(tuple(cur))
                return sorted(out)
            def slot_name(k_, end=False):
                if season_desc.startswith("time-of-day"):
                    s_ = ((k_ + (1 if end else 0)) * info["interval_s"]) % 86400
                    return f"{s_ // 3600:02d}:{(s_ % 3600) // 60:02d}"
                if season_desc == "day-of-week": return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k_ % 7]
                return f"slot {k_}"
            def win_name(s0, e0):
                if season_desc.startswith("time-of-day"): return f"{slot_name(s0)}–{slot_name(e0, end=True)}"
                return slot_name(s0) if s0 == e0 else f"{slot_name(s0)}–{slot_name(e0)}"
            wins = []
            for s0, e0 in runs(on):
                ks = np.arange(s0, e0 + 1) if e0 >= s0 else np.r_[np.arange(s0, n_lab), np.arange(0, e0 + 1)]
                m_d = (lab[idx_dev][:, None] == ks[None, :]).any(1); m_c = (lab[sel][:, None] == ks[None, :]).any(1)
                wins.append(dict(name=win_name(s0, e0), slots=len(ks), rate_dev=event[idx_dev][m_d].mean(), n_dev=int(m_d.sum()),
                                 rate_conf=event[sel][m_c].mean() if m_c.any() else float("nan"), n_conf=int(m_c.sum())))
            k_max = int(np.argmax(rate_sm)); k_min = int(np.argmin(rate_sm))
            sched = dict(rate_raw=rate_raw, rate_sm=rate_sm, rate_conf=rate_conf, on=on, w=w, wins=wins, any=bool(on.any()), all=bool(on.all()),
                         share_on=on.mean(), k_max=k_max, k_min=k_min, name_max=slot_name(k_max), name_min=slot_name(k_min),
                         slot_name=slot_name, cnt_c=cnt_c, cnt_d=cnt_d)
            pol["S"] = on[lab[sel]]
            dev_costs["S"] = cost(on[lab[idx_dev]], event[idx_dev])
        ex = event[sel]
        sw = {k: sw_of(v) for k, v in pol.items()}
        per_cluster = len(sel) / max(len(np.unique(cl[sel])), 1)
        res = {k: dict(cost=cost(v, ex, sw[k]), acts_per_cluster=v.mean() * per_cluster, switches_per_cluster=sw[k].mean() * per_cluster,
                       caught=int((ex & v).sum()), total=int(ex.sum())) for k, v in pol.items()}
        cls = cl[sel]
        def pc(k1, k2): return paired_clusters(cost, pol[k1], pol[k2], ex, cls, sw[k1], sw[k2])
        CB = pc("C", "B"); BpB = pc("B′", "B"); CBp = pc("C", "B′")
        cands = ["B′", "H"] + (["S"] if sched is not None and sched["any"] else []) + ["C"]      # from the simplest change to the most complex
        vsB = {k_: pc(k_, "B") for k_ in cands}
        pair = {(k2, k1): pc(k2, k1) for i_, k2 in enumerate(cands) for k1 in cands[:i_]}
        rel_conf = reliability(sel, p_ev)
        flag = sel[p_ev[sel] > P_STAR]
        flag_row = None
        if len(flag) > 2:
            g = gap_interval(flag, p_ev, event, cl)
            flag_row = dict(n=len(flag), share=len(flag) / len(sel), obs=event[flag].mean(), pred=p_ev[flag].mean(), gap=g[0], glo=g[1], ghi=g[2])
        skill_conf = 1 - ((x[sel] - fcast[sel]) ** 2).sum() / ((x[sel] - base_all[sel]) ** 2).sum()
        names = {"B′": f"the re-tuned threshold ({thB_star:.4g})", "H": f"the two-level rule (up {disp(H_up):.4g}, down {disp(H_dn):.4g})", "S": "the fixed schedule", "C": "the forecast policy"}
        beats = [k_ for k_ in cands if not isnan(vsB[k_]["hi"]) and vsB[k_]["hi"] < 0]
        losers = [k_ for k_ in cands if k_ not in beats]
        if not beats:
            action = ("Keep the existing rule. None of the candidates (" + ", ".join(names[k_] for k_ in cands) + ") beat it on the reserved period by a margin "
                      "whose interval excludes zero. Repeat the confirmation after more data with the same pre-specified evaluation.")
        else:
            choice = beats[0]; steps = []; undec = []
            for k_ in beats[1:]:
                pr_ = pair[(k_, choice)]
                if not isnan(pr_["hi"]) and pr_["hi"] < 0:
                    steps.append(f"{names[k_]} beats {names[choice]} with an interval that excludes zero"); choice = k_
                else:
                    undec.append(names[k_])
            action = f"Adopt {names[choice]}: its paired interval against the existing rule excludes zero."
            if steps: action += " " + "; ".join(steps) + ", so the more complex change is justified."
            if undec:
                u_ = ", ".join(undec[:-1]) + (" and " if len(undec) > 1 else "") + undec[-1]
                action += f" {u_[0].upper()}{u_[1:]} also beat{'s' if len(undec) == 1 else ''} the existing rule but {'is' if len(undec) == 1 else 'none is'} shown to beat {names[choice]} (undecided, not equivalent); the simpler change is preferred as a practical judgment."
            if losers: action += f" Not shown to beat the existing rule: {', '.join(names[k_] for k_ in losers)}; keep as candidates and re-confirm later."
            else: action += " Re-confirm after more data."
            if choice == "C": action += " The probability model needs periodic re-checking of its reliability."
        choice = beats and choice
        decision = dict(TAU=TAU, C_ACT=C_ACT, C_MISS=C_MISS, P_STAR=P_STAR, existing=a.existing, thB_star=thB_star, sigma=sigma, ev_desc=ev_desc,
                        rate_dev=event[dev].mean(), rate_conf=event[conf].mean(), rel_oof=rel_oof, n_oof=len(oof_idx), dev_costs=dev_costs, bins=bins,
                        res=res, CB=CB, BpB=BpB, CBp=CBp, vsB=vsB, pair=pair, cands=cands, names=names, beats=beats, choice=choice, sched=sched,
                        C_SW=C_SW, H=dict(up=H_up, down=H_dn, up_disp=disp(H_up), down_disp=disp(H_dn), desc=H_desc, same=H_same),
                        rel_conf=rel_conf, flag=flag_row, skill_conf=skill_conf, action=action, per_cluster=per_cluster,
                        _plot=dict(pol=pol, ex=ex, cls=cls, sw=sw))

    # ---- figures
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(np.arange(n), x, lw=0.5, color="#1f4e79")
    ax.axvline(split, color="#b30000", lw=1, ls="--"); ax.text(split, ax.get_ylim()[1], " reserved →", color="#b30000", va="top", fontsize=8)
    if decision:
        for lvl in ([decision["TAU"]] if a.direction != "abs" else [decision["TAU"], -decision["TAU"]]):
            ax.axhline(lvl, color="#888", lw=0.8, ls=":")
        ax.text(0, decision["TAU"], f" threshold {decision['TAU']:g}", va="bottom", fontsize=8, color="#555")
    ax.set_xlabel("row"); ax.set_ylabel(unit); ax.set_title("The series, with the reserved period marked", fontsize=10)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig_series = fig_to_b64(fig)
    fig_rel = fig_costs = None
    if decision:
        D = decision
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, rows, ttl in zip(axes, (D["rel_oof"], D["rel_conf"]), ("development, out of fold", "reserved period")):
            rows = [r for r in rows if r["n"] >= 20]
            top = max([r["pred"] for r in rows] + [r["obs"] for r in rows] + [D["P_STAR"]]) * 1.15
            ax.plot([0, top], [0, top], color="#aaa", lw=0.8)
            ax.axvline(D["P_STAR"], color="#b30000", lw=0.6, ls=":")
            for r in rows:
                err = [[r["gap"] - r["glo"]], [r["ghi"] - r["gap"]]] if not isnan(r["glo"]) else None
                ax.errorbar(r["pred"], r["obs"], yerr=err, fmt="o", color="#1f4e79", capsize=3, ms=5)
                ax.annotate(f"n={r['n']}", (r["pred"], r["obs"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#555")
            ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed rate")
            ax.set_title(f"reliability, {ttl} (bins with n ≥ 20; cluster 95%); dotted = break-even", fontsize=8.5)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        fig.tight_layout(); fig_rel = fig_to_b64(fig)
        fig_sched = None
        if D["sched"] is not None:
            Sd = D["sched"]; kk = np.arange(n_lab)
            fig, ax = plt.subplots(figsize=(10, 3.0))
            ax.bar(kk, 100 * Sd["rate_raw"], width=1.0, color="#c9d6e3", label="development, per slot")
            ax.plot(kk, 100 * Sd["rate_sm"], color="#1f4e79", lw=1.4, label=f"development, smoothed over {Sd['w']} slot{'s' if Sd['w'] > 1 else ''}")
            ax.plot(kk, 100 * Sd["rate_conf"], color="#b30000", lw=0.9, ls="--", label="reserved period, per slot (check)")
            ax.axhline(100 * D["P_STAR"], color="#b30000", lw=0.8, ls=":", label=f"break-even {100 * D['P_STAR']:.2g}%")
            on_ = Sd["on"]
            for k0 in kk[on_]: ax.axvspan(k0 - 0.5, k0 + 0.5, color="#ffe8b0", lw=0, zorder=0)
            nt = min(n_lab, 12); ticks = np.linspace(0, n_lab, nt, endpoint=False).astype(int)
            ax.set_xticks(ticks); ax.set_xticklabels([Sd["slot_name"](int(t_)) for t_ in ticks], fontsize=8)
            ax.set_xlim(-0.5, n_lab - 0.5); ax.set_ylabel("event rate, %"); ax.set_xlabel(season_desc)
            ax.set_title("Event rate by time of the cycle (shaded: slots where a fixed schedule would act)", fontsize=10)
            ax.legend(frameon=False, fontsize=7.5, ncol=2)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
            fig.tight_layout(); fig_sched = fig_to_b64(fig)
        fig, ax = plt.subplots(figsize=(10, 2.8))
        P = D["_plot"]; dd = np.unique(P["cls"])
        def cost(act, e, sw_): return D["C_ACT"] * act.mean() + D["C_MISS"] * (e & ~act).mean() + D["C_SW"] * sw_.mean()
        lines_ = [("C − B", "C", "B", "#1f4e79"), ("B′ − B", "B′", "B", "#b30000"), ("H − B", "H", "B", "#2e8b57")] + ([("S − B", "S", "B", "#c98a00")] if "S" in P["pol"] else [])
        for name, me, other, col in lines_:
            def cc(k_, m_): return cost(P["pol"][k_][m_], P["ex"][m_], P["sw"][k_][m_])
            diff = np.array([cc(me, P["cls"] == d) - cc(other, P["cls"] == d) for d in dd])
            ax.plot(np.cumsum(diff) / np.arange(1, len(dd) + 1), lw=1.2, color=col, label=name)
        ax.axhline(0, color="#888", lw=0.8); ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(f"cluster ({cluster_desc}) of the reserved period"); ax.set_ylabel("running mean cost diff. / step")
        ax.set_title("Running mean of the paired cluster cost differences", fontsize=10)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        fig_costs = fig_to_b64(fig)

    # ---- HTML
    d0, d1 = span(np.arange(n)); dv0, dv1 = span(dev); dc0, dc1 = span(conf)
    isec = info["interval_s"]
    interval_txt = ("not given; rows taken as evenly spaced" if not isec else "1 day" if isec == 86400 else f"{isec // 3600} h" if isec % 3600 == 0 and isec < 86400
                    else f"{isec // 60} min" if isec % 60 == 0 and isec < 3600 else f"{isec} s" if isec < 60 else f"{isec / 86400:g} days")
    checks = [("rows", f"{n:,}"), ("period", f"{d0} to {d1}"), ("sampling interval", interval_txt), ("transform", tr_desc),
              ("gaps / irregular steps", f"{info['gaps']}"), ("missing values", f"{info['missing']}"),
              ("timestamps strictly increasing", ("yes" if info["monotone"] else "NO") if t is not None else "no timestamps given"),
              ("known cycle used", season_desc), ("clusters for the intervals", cluster_desc),
              ("development", f"{dv0} to {dv1} ({len(dev):,} rows, {cl_dev} clusters)"),
              ("reserved", f"{dc0} to {dc1} ({len(conf):,} rows, {cl_conf} clusters), untouched until the final evaluation")]
    def table(head, rows, cls=""):
        hh = "".join(f"<th>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f'<table class="{cls}"><thead><tr>{hh}</tr></thead><tbody>{b}</tbody></table>'
    def ci(d): return "—" if isnan(d["lo"]) else f"<span class='nw'>[{d['lo']:+.4f}, {d['hi']:+.4f}]</span>"
    def nw(d): return "—" if isnan(d["nw"][0]) else f"<span class='nw'>[{d['nw'][0]:+.4f}, {d['nw'][1]:+.4f}]</span>"
    H = []
    _sec = [0]
    def h2(t_):
        _sec[0] += 1
        return f"<h2>{_sec[0]}. {t_}</h2>"
    H.append(f"<h1>{title}</h1><p class='meta'>Generated {time.strftime('%Y-%m-%d %H:%M')} by sequence_report.py from <code>{os.path.basename(a.csv)}</code> (column <code>{vcol}</code>"
             + (f", timestamps <code>{tcol}</code>" if tcol else "") + f"). Seed {a.seed}. Every number below is reproducible from the CSV, this script and the settings listed at the end.</p>")

    # plain-language summary
    D = decision
    what = "days" if cluster_desc.startswith("calendar") else "blocks"
    summ = [f"<b>What was analysed.</b> {n:,} values of <i>{unit}</i>" + (f", one every {interval_txt}" if isec else "") + f", from {d0} to {d1}. "
            f"The last {cl_conf} {what} ({len(conf):,} values) were locked away before any analysis and used only once, at the end, "
            "to check whether anything found earlier holds up on data it had never seen."]
    if lab is not None:
        summ.append(f"<b>The obvious pattern.</b> The value follows a regular cycle ({season_desc}). That cycle is treated as known and removed first, so everything below is about what remains <i>after</i> the cycle.")
    if levelish:
        summ.append(f"<b>Warning: this looks like a level, not a signal.</b> Simply repeating the last value already explains {100 * ev['persistence']:.0f}% of the variation, "
                    "which is what a cumulative or slowly wandering quantity (a price, a balance, a running total, a temperature) does. Against the typical-value baseline every model looks brilliant and none of it is predictive. "
                    "Re-run with <code>--transform diff</code> (or <code>pct</code> / <code>logdiff</code>) to analyse the <i>changes</i> from one step to the next; the rest of this report should be read with that in mind.")
    elif usable:
        tail = f", and again on the locked-away data ({100 * D['skill_conf']:.0f}% better)." if decision else "."
        summ.append(f"<b>Is there a pattern beyond that?</b> Yes. The recent past says something about the value {h} step{'' if h == 1 else 's'} ahead: a simple model that looks at the last {a.lags} values predicts it "
                    f"{100 * ev['skill']:.0f}% better (in squared error) than just using the typical value for that time, and it did so consistently in every stretch of the data" + tail)
    elif pattern:
        summ.append(f"<b>Is there a pattern beyond that?</b> Detectable, but not usable. With this much data the tests can see faint traces of order in the values, "
                    f"yet a simple model that looks at the last {a.lags} values predicts the value {h} step{'' if h == 1 else 's'} ahead "
                    + ("no better than the typical value for that time" if ev["skill"] < a.min_skill else "better on average but not in every stretch of the data")
                    + f" (skill {ev['skill']:+.3f}; a flexible forecaster scored {fc[0]:+.3f}). Statistically there is something; practically there is nothing to forecast with these tools.")
    else:
        summ.append(f"<b>Is there a pattern beyond that?</b> Nothing detectable with these checks: the recent past does not help predict the value {h} step{'' if h == 1 else 's'} ahead better than the typical value for that time.")
    if decision and D["sched"] is not None:
        Sd = D["sched"]
        if Sd["all"]:
            summ.append(f"<b>Does the risk depend on the time of the cycle?</b> No: the event rate is above the break-even chance ({100 * D['P_STAR']:.2g}%) at every time of the cycle "
                        f"(lowest around {Sd['name_min']}, {100 * Sd['rate_sm'][Sd['k_min']]:.1f}%; highest around {Sd['name_max']}, {100 * Sd['rate_sm'][Sd['k_max']]:.1f}%), so a fixed schedule would act all the time.")
        elif Sd["any"]:
            summ.append(f"<b>Does the risk depend on the time of the cycle?</b> Yes. On the development data the event rate rises above the break-even chance ({100 * D['P_STAR']:.2g}%) only in "
                        + "; ".join(f"{w_['name']} ({100 * w_['rate_dev']:.1f}%" + (f", locked-away data {100 * w_['rate_conf']:.1f}%" if not isnan(w_["rate_conf"]) else "") + ")" for w_ in Sd["wins"])
                        + f" — {100 * Sd['share_on']:.0f}% of the cycle — with the highest risk around {Sd['name_max']} ({100 * Sd['rate_sm'][Sd['k_max']]:.1f}%) and the lowest around {Sd['name_min']} ({100 * Sd['rate_sm'][Sd['k_min']]:.1f}%). "
                        f"A fixed schedule that adds capacity only in those windows is therefore a candidate: on the locked-away data it cost {D['res']['S']['cost']:.4g} per step "
                        f"({pct(abs(D['vsB']['S']['mean']) / D['res']['B']['cost'])} {'less' if D['vsB']['S']['mean'] < 0 else 'more'} than the current rule) and handled {D['res']['S']['caught']} of {D['res']['S']['total']} events.")
        else:
            summ.append(f"<b>Does the risk depend on the time of the cycle?</b> Not enough to schedule by: at no time of the cycle does the event rate exceed the break-even chance ({100 * D['P_STAR']:.2g}%) on the development data "
                        f"(highest around {Sd['name_max']}, {100 * Sd['rate_sm'][Sd['k_max']]:.1f}%; lowest around {Sd['name_min']}, {100 * Sd['rate_sm'][Sd['k_min']]:.1f}%). "
                        "A fixed schedule alone would never act, so the events have to be caught by a rule or a forecast.")
    if decision:
        summ.append(f"<b>Is it worth acting on?</b> That was tested on the locked-away data with the costs attached (acting costs {D['C_ACT']:g}, a missed event costs {D['C_MISS']:g}; the event is: {D['ev_desc']}). "
                    f"The current rule cost {D['res']['B']['cost']:.4g} per step. Using the forecast cost {D['res']['C']['cost']:.4g} ({pct(abs(D['CB']['mean']) / D['res']['B']['cost'])} {'less' if D['CB']['mean'] < 0 else 'more'}), "
                    f"and simply moving the current rule's trigger level from {D['existing']:g} to {D['thB_star']:.4g} cost {D['res']['B′']['cost']:.4g} ({pct(abs(D['BpB']['mean']) / D['res']['B']['cost'])} {'less' if D['BpB']['mean'] < 0 else 'more'}). "
                    "Each difference comes with a range of plausible values; a saving counts only if its whole range is on the saving side.")
        Hd = D["H"]; HBp = D["pair"][("H", "B′")]
        gain_txt = (f"On the locked-away data it cost {D['res']['H']['cost']:.4g} per step against {D['res']['B′']['cost']:.4g} for the single threshold "
                    f"({pct(abs(HBp['mean']) / D['res']['B′']['cost'])} {'less' if HBp['mean'] < 0 else 'more'}, plausible range {ci(HBp)}), "
                    + ("so the lower release level is a real saving." if not isnan(HBp["hi"]) and HBp["hi"] < 0 else "so no saving from the lower release level is shown."))
        if D["C_SW"] == 0:
            summ.append(f"<b>When to scale back down?</b> This report assumes a change of capacity costs nothing (<code>--cost-switch 0</code>). Under that assumption capacity can be released as soon as the trigger condition no longer holds; "
                        f"a separate, lower release level only pays when each change costs something. Checked anyway: the best two-level rule found on the development data would {Hd['desc']}"
                        + (" — the same level both ways, i.e. no separate release level helps here. " if Hd["same"] else ". ") + gain_txt
                        + " If scaling up or down does cost something (start-up time, migration, a fee), re-run with <code>--cost-switch</code> and this comparison becomes the deciding one.")
        else:
            summ.append(f"<b>When to scale back down?</b> Each change of capacity costs {D['C_SW']:g}, so flapping up and down is penalised. The best two-level rule found on the development data would {Hd['desc']}"
                        + (" — the same level both ways. " if Hd["same"] else ". ") + gain_txt
                        + f" For reference, the existing rule changed state {D['res']['B']['switches_per_cluster']:.1f} times per {what[:-1]} on the locked-away data, the two-level rule {D['res']['H']['switches_per_cluster']:.1f} times.")
        summ.append(f"<b>What to do.</b> {D['action']}")
    else:
        summ.append("<b>Is it worth acting on?</b> Not tested: no event, threshold or costs were supplied. A pattern on its own is not a reason to change anything.")
    H.append("<div class='box'><p class='boxtitle'>Summary in plain words</p>" + "".join(f"<p>{p_}</p>" for p_ in summ) + "</div>")

    H.append(h2("Series and data checks") + table(["check", "result"], checks))
    if info["notes"]:
        H.append("<ul>" + "".join(f"<li>{s_}</li>" for s_ in info["notes"]) + "</ul>")
    H.append("<p class='plain'><b>In plain words.</b> Before any statistics, the data itself is checked: are the readings evenly spaced in time, are any missing, is anything out of order? "
             "Each of those creates false patterns. \"Development\" is the part of the data used to look for patterns and fit models; \"reserved\" is the part kept back so that whatever is found can be tested on data that played no part in finding it. "
             "Without that split, almost anything looks like a pattern. \"Clusters\" are the blocks (usually days) treated as independent when the uncertainty of a result is computed.</p>")
    H.append(f'<img src="data:image/png;base64,{fig_series}">')
    if decision:
        H.append(h2("Decision") +
                 f"<p>At each step, act for the step <b>{h}</b> ahead if the event — <b>{D['ev_desc']}</b> — is likely. Acting costs {D['C_ACT']:g}; an unhandled event costs {D['C_MISS']:g}; "
                 f"so acting pays when the event probability exceeds {D['C_ACT']:g}/{D['C_MISS']:g} = <b>{100 * D['P_STAR']:.2g}%</b>. "
                 f"<b>Existing rule (B):</b> act when the current value is {'above' if a.direction == 'above' else 'below' if a.direction == 'below' else 'beyond ±'} {D['existing']:g}. "
                 f"<b>Proposed (C):</b> act when the forecast event probability exceeds {100 * D['P_STAR']:.2g}%. "
                 f"References: never act (A); the existing rule re-tuned on the development period (B′, threshold {D['thB_star']:.4g}); "
                 f"a two-level rule (H) re-tuned on the development period with a separate release level: {D['H']['desc']}"
                 + (f" (the same level both ways, so H coincides with B′)" if D['H']['same'] else "") + f". Each change of state costs {D['C_SW']:g}"
                 + (" (changes are free; a release level below the trigger can then only help by chance)." if D["C_SW"] == 0 else " (so every scale-up and every scale-down is charged).") + "</p>"
                 "<p class='small'>Assumed: acting prevents the whole penalty for that step, is available in time, carries no start-up or switching cost, and is paid for whether or not the event occurs; each step's decision stands alone. "
                 f"Event rate: development {pct(D['rate_dev'])}, reserved {pct(D['rate_conf'])}.</p>")
        H.append("<p class='plain'><b>In plain words.</b> The decision is a bet. Acting costs a little every time; not acting costs a lot when the event happens. Dividing the two costs gives the break-even chance: "
                 f"act whenever the event is more likely than {100 * D['P_STAR']:.2g}%. Four ways of deciding are compared: do nothing (A), the rule in use today (B), the same rule with its trigger level re-chosen on past data (B′), and acting on a forecast (C). "
                 "B′ is there to answer a fair question: is a forecast better than merely adjusting the number in the existing rule?</p>")
    else:
        H.append(h2("Decision") + "<p>No decision was specified (pass <code>--threshold</code>, <code>--existing</code> and <code>--cost-miss</code>, and <code>--direction</code> if the event is not \"above\"). "
                 "The report therefore stops at detection: whether structure exists, not whether it is worth acting on.</p>")
    H.append(h2(f"Known structure and baselines (development, rolling origin, {a.folds} folds, horizon {h})"))
    base_rows = [(f"linear AR({a.lags}) on the deseasonalised value, vs the calendar-mean baseline", f3(ev["skill"]) + "  (per fold: " + ", ".join(f"{s_:+.2f}" for s_ in ev["per_fold"]) + ")"),
                 ("persistence (last value known at decision time)", f3(ev["persistence"]))]
    if ev["seasonal_persistence"] is not None:
        base_rows.append(("seasonal persistence (same slot one cycle earlier)", f3(ev["seasonal_persistence"])))
    H.append(table(["baseline / model", "skill (1 − MSE/MSE_baseline)"], base_rows))
    if levelish:
        H.append(f"<p class='small'><b>Baseline warning.</b> Persistence scores above {a.level_warn:g}: the series behaves like a level that wanders, for which the calendar-mean baseline is the wrong reference and any model's skill against it is inflated. "
                 "Re-run with <code>--transform diff</code> (or <code>pct</code>, <code>logdiff</code>); the question of interest is whether the <i>changes</i> are predictable.</p>")
    H.append(f"<p class='small'>Calendar means ({season_desc}) and coefficients are estimated inside each training window, never from the fold being scored. "
             + ("All folds positive: the stability check passes." if stable else "<b>Not all folds are positive</b>: the skill is not stable across the development period.") + "</p>")
    H.append("<p class='plain'><b>In plain words.</b> \"Skill\" measures how much better a forecast is than a naive one; 0 means no better, positive means better, negative means worse. The naive forecast here is the typical value for that time of the cycle. "
             f"\"Persistence\" (assume the future value equals the latest known one) scores {ev['persistence']:+.2f}" + (f", and \"seasonal persistence\" (assume it equals the value one cycle ago) scores {ev['seasonal_persistence']:+.2f}" if ev['seasonal_persistence'] is not None else "") +
             f". The simple model scores {ev['skill']:+.2f}. \"Rolling origin\" means the model was always fitted on earlier data and scored on later data, {a.folds} times over successive stretches (the numbers in brackets); "
             "if they are all positive the improvement is steady rather than a lucky patch.</p>")
    H.append(h2("Diagnostics (development, deseasonalised)") + table(["diagnostic", "statistic", "p", "null", "reading"], diagnostics))
    H.append("<p class='small'>Read each p-value against its own null. Rejections say the sequence is not exchangeable / not stationary / informative about its future; none says the information is worth acting on. "
             "Non-rejection by the change tests does not establish stationarity. The scale test is exploratory under persistent variance clustering (size about twice nominal).</p>")
    H.append("<p class='plain'><b>In plain words.</b> Each diagnostic asks one question and answers it with a p-value: the chance of seeing a result at least this strong if the answer were \"no\". "
             f"A p-value below 0.05 is the usual bar for \"this is unlikely to be chance\"; \"p ≤ {1 / (lb[2] + 1):.3f}\" means the result was stronger than every one of the {lb[2]} shuffled versions of the data the test compared it with. "
             "The first two rows ask whether the order of the values matters (Ljung–Box: do neighbouring values move together? kNN forecast: does the recent past help predict the future, beyond the cycle?). "
             "The last two rows ask whether the series changed character part-way through, in its typical level or in how much it varies; a change like that would make any pattern found earlier untrustworthy. "
             "\"No level shift detected\" means the test did not find one, not that none exists.</p>")
    if decision:
        H.append(h2("Probability check") +
                 f"<p>Forecast {h} step{'' if h == 1 else 's'} ahead = calendar mean + AR({a.lags}) prediction; residual spread on development {D['sigma']:.4g}; event probability from the Gaussian tail ({D['ev_desc']}). "
                 f"Reliability is checked <b>out of fold</b> on development ({D['n_oof']:,} steps whose probability came from a model that never saw them) and on the reserved period. "
                 f"Bins are built around the break-even probability ({100 * D['P_STAR']:.2g}%): half of it, it, twice, four times. "
                 f"Intervals are cluster-robust ({cluster_desc}: sums of outcome − predicted probability per cluster, variance across clusters; clusters treated as independent), which allows for dependence between consecutive predictions and unequal probabilities inside a bin.</p>")
        rr = lambda rows: [(f"[{100 * r['lo']:.3g}%, {100 * r['hi']:.3g}%{')' if r['hi'] < 1 else ']'}", f"{r['n']:,}", pct(r["pred"]), pct(r["obs"]), f"{pts(r['gap'])} pts", ("—" if isnan(r["glo"]) else f"[{pts(r['glo'])}, {pts(r['ghi'])}]")) for r in rows]
        H.append("<p class='cap'>Development, out of fold</p>" + table(["bin", "n", "mean predicted", "observed", "observed − predicted", "cluster 95%"], rr(D["rel_oof"])))
        H.append("<p class='cap'>Reserved period (model frozen at the split)</p>" + table(["bin", "n", "mean predicted", "observed", "observed − predicted", "cluster 95%"], rr(D["rel_conf"])))
        if D["flag"]:
            f_ = D["flag"]
            H.append(f"<p class='small'>Steps flagged above {100 * D['P_STAR']:.2g}% on the reserved period: {f_['n']:,} ({pct(f_['share'])}); observed event rate among them {pct(f_['obs'])} against a mean prediction of {pct(f_['pred'])}; "
                     f"observed − predicted {pts(f_['gap'])} pts, cluster 95% [{pts(f_['glo'])}, {pts(f_['ghi'])}]. Forecast skill on the reserved period, frozen at the split: {D['skill_conf']:+.3f}.</p>")
        H.append(f'<img src="data:image/png;base64,{fig_rel}">')
        H.append("<p class='plain'><b>In plain words.</b> The forecast does not say \"an event will happen\"; it gives a chance. This section checks whether those chances are honest: among all the steps where the forecast said \"about 7%\", did an event actually happen about 7% of the time? "
                 "Each row groups steps by their predicted chance and compares the prediction with what happened. \"Observed − predicted\" near zero means honest; the range next to it says how far off the truth could plausibly be given how few steps the row holds. "
                 "The rows that matter for the decision are the ones above the break-even chance, and they are usually the thinnest.</p>")
        if D["sched"] is not None:
            Sd = D["sched"]
            H.append(h2("Event risk by time of the cycle (development)"))
            H.append(f"<p>Event rate per slot of the cycle ({season_desc}) on the development period, smoothed over {Sd['w']} neighbouring slot{'s' if Sd['w'] > 1 else ''}; "
                     f"a fixed schedule <b>S</b> acts in every slot whose smoothed rate exceeds the break-even chance ({100 * D['P_STAR']:.2g}%). "
                     f"Highest risk around {Sd['name_max']} ({100 * Sd['rate_sm'][Sd['k_max']]:.1f}%), lowest around {Sd['name_min']} ({100 * Sd['rate_sm'][Sd['k_min']]:.1f}%); "
                     f"the schedule covers {100 * Sd['share_on']:.0f}% of the cycle. The development period holds {int(event[idx_dev].sum()):,} events spread over {n_lab} slots "
                     f"({event[idx_dev].sum() / n_lab:.1f} per slot on average), so per-slot rates are noisy; the smoothing and the reserved-period check guard against reading noise as timing. "
                     "The reserved-period rates are shown only as a check; the schedule was fixed before they were looked at.</p>")
            if Sd["wins"]:
                H.append(table(["window (schedule acts)", "slots", "development event rate", "n", "reserved event rate (check)", "n"],
                               [(w_["name"], f"{w_['slots']}", pct(w_["rate_dev"]), f"{w_['n_dev']:,}", pct(w_["rate_conf"]) if not isnan(w_["rate_conf"]) else "—", f"{w_['n_conf']:,}") for w_ in Sd["wins"]]))
            else:
                H.append("<p class='small'>No window: the smoothed rate stays below the break-even chance everywhere, so S never acts (identical to A) and is not a candidate.</p>")
            H.append(f'<img src="data:image/png;base64,{fig_sched}">')
            H.append("<p class='plain'><b>In plain words.</b> This asks a simpler question than forecasting: is the trouble tied to the clock? If events pile up in the same part of the day (or week) every cycle, then adding capacity on a fixed timetable — "
                     "no forecast, no threshold, just \"more from 11:00 to 17:00\" — is a candidate policy, and it is evaluated below alongside the others on the locked-away data. "
                     "The shaded slots are where the schedule would act; the dashed line is what actually happened in the locked-away period, a check that the timing held up. "
                     "Rates in the ups and downs of the cycle that never reach the break-even chance do not justify scheduled capacity on their own, however visible the cycle is.</p>")
        H.append(h2("Decision evaluation (reserved period; everything fixed from development)"))
        pr = D["res"]; per_lbl = 'day' if cluster_desc.startswith('calendar') else 'cluster'
        labels = {"A": "A never act", "B": f"B existing rule ({D['existing']:g})", "B′": f"B′ re-tuned rule ({D['thB_star']:.4g})",
                  "H": f"H two-level rule (up {D['H']['up_disp']:.4g}, down {D['H']['down_disp']:.4g})", "C": f"C forecast probability > {100 * D['P_STAR']:.2g}%"}
        if "S" in pr: labels["S"] = f"S fixed schedule ({100 * D['sched']['share_on']:.0f}% of the cycle)"
        order = [k_ for k_ in ("A", "B", "B′", "H", "S", "C") if k_ in pr]
        H.append(table(["policy", "cost / step", f"actions per {per_lbl}", f"changes of state per {per_lbl}", "events handled"],
                       [(labels[k_], f"{pr[k_]['cost']:.4g}", f"{pr[k_]['acts_per_cluster']:.2f}", f"{pr[k_]['switches_per_cluster']:.2f}", f"{pr[k_]['caught']} / {pr[k_]['total']}") for k_ in order]))
        dc = D["dev_costs"]
        H.append("<p class='small'>Development costs per step, for reference (in-sample for the fitted model): " + ", ".join(f"{k_} {dc[k_]:.4g}" for k_ in ("A", "B", "Bp", "H", "S", "C") if k_ in dc).replace("Bp ", "B′ ") + ".</p>")
        tag = {"B′": " (the threshold change itself)", "H": " (the release level added)", "S": " (the schedule against the existing rule)", "C": ""}
        rows_ = [(f"{k_} − B{tag[k_]}", f"{D['vsB'][k_]['mean']:+.4g}", pct(D['vsB'][k_]['mean'] / pr['B']['cost']), ci(D['vsB'][k_]), f3(D['vsB'][k_]['rho1']), nw(D['vsB'][k_])) for k_ in D["cands"]]
        for (k2, k1), pr_ in D["pair"].items():
            rows_.append((f"{k2} − {k1}" + (" (is the release level worth anything?)" if (k2, k1) == ("H", "B′") else ""), f"{pr_['mean']:+.4g}", pct(pr_['mean'] / pr[k1]['cost']), ci(pr_), f3(pr_['rho1']), nw(pr_)))
        H.append(table(["comparison", "mean cost difference / step", "relative to the second policy", "95% t-interval (paired clusters)", "lag-1 autocorr. of cluster diffs", "Newey–West (2 lags)"], rows_))
        H.append(f"<p class='small'>Paired differences over {D['CB']['D']} reserved clusters ({cluster_desc}); clusters are treated as independent (check the lag-1 autocorrelation and the Newey–West interval, which adjusts for serial dependence). "
                 "A negative difference favours the first policy. An interval that includes zero means undecided, not equivalent.</p>")
        H.append(f'<img src="data:image/png;base64,{fig_costs}">')
        H.append("<p class='plain'><b>In plain words.</b> Each policy was run on the locked-away data exactly as it would have been run live, with nothing adjusted, and its total bill divided by the number of steps. "
                 "\"Actions\" is how often it would have acted; \"changes of state\" how often it switched between acting and not acting (each one is charged when a switching cost is given); \"events handled\" how many of the events it would have acted on in time. "
                 "The second table asks whether the differences between policies are real or luck: the bill is compared cluster by cluster (usually day by day), and the range shows where the true average difference plausibly lies. "
                 "A negative number favours the first-named policy. If the range straddles zero, the data were too few or too variable to tell the two policies apart, which is not the same as saying they are equal. "
                 "The last two columns check that consecutive clusters are not moving together; the Newey–West range corrects for that and should be close to the first.</p>")
        H.append(h2("Reading and action"))
        struct = ("Structure detected on the development period: " + "; ".join(d_[4] for d_, hit in zip(diagnostics[:2], (lb[1] < 0.05, fc[1] < 0.05)) if hit)) if pattern \
            else "No structure detected on the development period by the two serial diagnostics."
        H.append(f"<p><b>Structure.</b> {struct}. Rolling-origin skill {ev['skill']:+.3f}, " + ("stable across folds" if stable else "not stable across folds") +
                 f"; on the reserved period, frozen at the split, {D['skill_conf']:+.3f}." +
                 ("" if ev["skill"] >= a.min_skill else f" <b>Detection is not improvement</b>: the diagnostics see faint order, but no forecaster here beats the calendar-mean baseline by the usable margin ({a.min_skill:g}).") + "</p>")
        H.append(f"<p><b>Value.</b> {D['action']}</p>")
    else:
        H.append(h2("Reading"))
        struct = ("Structure detected: " + "; ".join(d_[4] for d_, hit in zip(diagnostics[:2], (lb[1] < 0.05, fc[1] < 0.05)) if hit)) if pattern else "No structure detected by the two serial diagnostics at this n."
        H.append(f"<p>{struct}. Rolling-origin skill {ev['skill']:+.3f}, " + ("stable across folds." if stable else "not stable across folds.") +
                 ("" if ev["skill"] >= a.min_skill else f" <b>Detection is not improvement</b>: no forecaster here beats the calendar-mean baseline by the usable margin ({a.min_skill:g}).") +
                 " Whether any of it is worth acting on needs a decision, costs and a reserved-period evaluation; re-run with <code>--threshold</code>, <code>--existing</code> and <code>--cost-miss</code>.</p>")
    rules = [("a pattern counts as usable", f"rolling-origin skill ≥ {a.min_skill:g} and positive in every fold, and the series is not level-like"),
             ("a fixed schedule (S) acts", "in every slot of the cycle whose development event rate, smoothed over neighbouring slots, exceeds the break-even chance; it is a candidate only if it acts somewhere"),
             ("the two-level rule (H)", "both levels chosen on the development data over a 61 × 61 quantile grid (release level ≤ trigger level), with the switching cost included; it is a candidate even when it coincides with B′"),
             ("candidates are ordered by simplicity", "re-tuned threshold, then two-level rule, then fixed schedule, then forecast policy; a more complex candidate is chosen only if it beats the simpler one"),
             ("a candidate beats a policy", "its paired interval against that policy lies entirely below zero"),
             ("two candidates the data cannot separate", "the simpler one is preferred as a practical judgment (undecided ≠ equivalent)"),
             ("a detected pattern on its own", "is never a reason to change the policy")]
    H.append("<h2>Rules the reading follows</h2>" + table(["rule", "as applied"], rules))
    settings = [("reserve", f"{a.reserve:g}"), ("lags / horizon", f"{a.lags} / {h}"), ("folds / first training window", f"{a.folds} / {a.first_train:g}"), ("kNN neighbours", f"{a.k}"),
                ("period", "inferred" if a.period is None else str(a.period)), ("cluster", cluster_desc), ("transform", a.transform), ("direction", a.direction if decision else "—"),
                ("costs act / miss / switch", f"{a.cost_act:g} / {a.cost_miss if a.cost_miss is not None else '—'} / {a.cost_switch:g}"), ("min-skill / level-warn", f"{a.min_skill:g} / {a.level_warn:g}"), ("B (Ljung–Box / forecast)", f"{a.B_lb} / {a.B_forecast}"), ("seed", f"{a.seed}"), ("delimiter", repr(sep))]
    H.append("<h2>Settings used</h2>" + table(["setting", "value"], settings))
    gl = [
        ("rows / period / sampling interval", "how many values there are, the dates they cover, and how far apart in time consecutive values are."),
        ("transform", "whether the values were analysed as given (levels) or as changes from one step to the next; changes are the right input for a quantity that wanders, such as a price or a running total."),
        ("gaps, missing values, strictly increasing", "problems in the raw data that would create false patterns: skipped or duplicated times, empty cells, values out of order."),
        ("development / reserved", "the data used to search and fit (development) and the data locked away to test the result once (reserved)."),
        ("clusters", "blocks of consecutive values (usually calendar days) treated as independent when uncertainty is computed; values inside a cluster may be related."),
        ("known cycle", "a regular daily or weekly rhythm inferred from the timestamps. It is removed before testing, because a rhythm everyone already knows about is not a discovery."),
        ("calendar mean / baseline", "the typical value for that time of the cycle, learned from earlier data only. Every forecast is judged against it."),
        ("horizon", "how many steps ahead the forecast and the decision look."),
        ("skill", "1 − (the model's squared error ÷ the baseline's squared error). 0 = no better than the baseline; +0.10 = 10% less squared error; negative = worse."),
        ("persistence / seasonal persistence", "two naive forecasts: \"the future value equals the latest known one\" and \"it equals the value one cycle ago\". Any proposed model must beat them."),
        ("rolling origin, per fold", "the model is fitted on earlier data and scored on later data, repeatedly, moving forward in time; each stretch is a fold. Consistent positive folds mean a steady effect."),
        ("Ljung–Box Q(10)", "a single number summarising how strongly each value is related to the previous ten. Larger means stronger relation."),
        ("kNN forecast skill", "the skill of a simple, flexible forecaster (k nearest neighbours) that uses the last few values; it can pick up patterns a straight-line model misses."),
        ("null / permutation / B", "the null is the \"nothing here\" scenario the test compares against, made concrete by shuffling the data B times in a way that destroys exactly the pattern in question and keeps everything else."),
        ("p-value", "the share of shuffles (or, for analytic tests, the theoretical chance) that did as well as the real data. Below 0.05 is the customary bar; p ≤ 1/(B+1) is the smallest value B shuffles can report."),
        ("CUSUM level / scale", "tests for a change part-way through the series in its typical level, or in how widely it varies. \"Exploratory\" on the scale test means its p-values are known to be somewhat too small on series whose variability comes in bursts."),
        ("event / threshold / direction", "what the decision cares about: the value going above, below, or beyond ± the threshold."),
        ("residual spread", "how far the forecast typically misses by. Together with the forecast it gives the chance of an event."),
        ("bins, observed − predicted, cluster 95%", "steps grouped by predicted chance (around the break-even chance); the difference between what happened and what was predicted; and the range in which that difference plausibly lies, computed cluster by cluster."),
        ("event risk by time of the cycle / schedule (S)", "how often the event happens in each slot of the daily or weekly rhythm, learned on the development data; the schedule adds capacity in the slots where that rate is above the break-even chance, with no forecast involved."),
        ("two-level rule (H) / release level / switching cost", "scale up when the value passes the trigger level, and scale back down only when it falls below a lower release level; the gap stops the system flapping. The switching cost is what each change costs (start-up, migration, a fee); with a switching cost of zero a lower release level can only help by chance."),
        ("policies A, B, B′, C", "never act; the existing rule; the existing rule with its trigger level re-chosen on development data; act when the forecast chance exceeds the break-even chance."),
        ("break-even chance", "cost of acting ÷ cost of a missed event. Acting is worth it when the chance of the event is above this."),
        ("cost / step, actions, events handled", "the average bill per step under a policy; how often it acts; how many events it acted on in time."),
        ("paired difference, 95% t-interval", "the bill of one policy minus the bill of another, computed for each cluster, then averaged; the interval is the range of plausible averages. If it straddles zero the data cannot say which policy is cheaper."),
        ("lag-1 autocorrelation, Newey–West", "a check that consecutive clusters are not moving together (which would make the interval too narrow); the Newey–West interval adjusts for that and should be close to the plain one."),
        ("seed", "the starting point of the random shuffles, fixed so that re-running gives the same numbers."),
    ]
    H.append("<h2>What each number means</h2>" + table(["term", "meaning"], gl, cls="gloss"))
    H.append(f"<p class='meta'>Runtime {time.time() - t0:.0f} s. Diagnostics from randomness_toolkit.py; method and validation in the accompanying article.</p>")

    css = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; @bottom-center { content: counter(page) ' / ' counter(pages); font-family: Charter, Georgia, serif; font-size: 8.5pt; color: #666; } }
body { font-family: "Charter", "Georgia", "Times New Roman", serif; font-size: 10pt; line-height: 1.42; color: #111; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; page-break-after: avoid; }
p { margin: 5pt 0; } .meta { color: #666; font-size: 8.5pt; } .small { font-size: 9pt; color: #333; } .cap { font-size: 9pt; font-weight: bold; margin: 8pt 0 2pt; }
code { font-family: Menlo, "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
table { border-collapse: collapse; font-size: 8.4pt; margin: 4pt 0 8pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; } th { background: #eee; } .nw { white-space: nowrap; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
.box { border: 1px solid #999; background: #f6f6f2; padding: 6pt 10pt; margin: 8pt 0 10pt; page-break-inside: avoid; }
.boxtitle { font-weight: bold; font-size: 11pt; margin: 0 0 4pt; }
.plain { font-size: 9.3pt; color: #222; margin: 4pt 0 8pt; padding-left: 8pt; border-left: 2px solid #bbb; }
table.gloss td:first-child { width: 28%; font-weight: bold; }
"""
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{''.join(H)}</body></html>"
    out = pathlib.Path(a.out)
    html_path = out.with_suffix(".html"); html_path.write_text(html)
    chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium",
                               shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("chrome")] if c and os.path.exists(c)), None)
    if chrome and out.suffix.lower() == ".pdf":
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline", "--no-margins",
                        f"--print-to-pdf={out.resolve()}", html_path.resolve().as_uri()], check=True, capture_output=True, timeout=180)
        print(f"wrote {out} ({out.stat().st_size:,} bytes) and {html_path}; {time.time() - t0:.0f} s")
    else:
        print(f"no Chrome/Chromium found: wrote {html_path} (open it in a browser and print to PDF); {time.time() - t0:.0f} s")
    if decision:
        keep = {k: v for k, v in decision.items() if k not in ("_plot", "names")}
        keep["pair"] = {f"{k_[0]} - {k_[1]}": v for k_, v in decision["pair"].items()}
        if decision["sched"] is not None:
            Sd = decision["sched"]
            keep["sched"] = dict(w=Sd["w"], windows=Sd["wins"], share_on=Sd["share_on"], any=Sd["any"], all=Sd["all"], slot_max=Sd["name_max"], slot_min=Sd["name_min"],
                                 rate_dev_smoothed=[float(v) for v in Sd["rate_sm"]], rate_dev_raw=[float(v) for v in Sd["rate_raw"]], rate_reserved=[float(v) for v in Sd["rate_conf"]])
        json.dump(keep, open(out.with_suffix(".json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()
