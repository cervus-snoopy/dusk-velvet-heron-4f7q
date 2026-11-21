#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a target instant.

Model: earner COUNTS and tracked PLAYERS, fitted separately in log
space, rate = earners / players. Everything is derived from the case
file at run time; nothing is read from the network.

  * Time is u = ln(1 + hours since launch, 2026-11-19T00:00:00Z).
    Launch-week growth saturates in u, so a straight line in u is a
    conservative extrapolation over a multi-day horizon.
  * Players P(t): least-squares line of ln(P) on u over the case's
    reads. P only grows (the tracked set never shrinks), so the
    forecast is floored at the last read and capped at 12x it.
  * Earners E_i(t): least-squares line of ln(max(E_i, 1)) on u per
    trophy, where E_i = round(rate_i * P / 100) is the earner count
    implied by the printed rate. Earners are monotone (nobody
    un-earns a trophy), so the forecast is floored at the last
    observed count, capped at 12x it and at P(target).
  * Flat history (the page has not refreshed yet): the next refresh
    dilutes rates, because new tracked players have earned far fewer
    trophies. The rules state the launch-night refresh outright:
    638 players tracked became 3,868, a factor of 3868/638, while
    every rate but the one at 100% fell. So a flat history forecasts
    one such refresh (P x 3868/638, earners constant) for a short
    horizon, and a milder continued growth for a long one.
  * The 100% trophy: the first refresh left it at 100% while every
    other rate fell (new players earn it immediately), so its median
    stays at 99.99 with a tight band.
  * Spread on the logit scale combines a random-walk term that grows
    with the horizon (0.5 logit per sqrt day, the house baseline's
    own number) with the residual scatter of the earner fit and the
    player fit, because the rate's log is ln(E) - ln(P).
  * Quantiles are rounded to 2 decimals: Exophase prints rates to 2
    decimals, so the answering rate is a 2-decimal number and a
    2-decimal median is the closest honest centre to it.

Standard library only. Deterministic: fixed iteration order, no
randomness, no wall-clock use. Reads every *.in file in
$MAKEFUN_INPUT_DIR (else /input/) and writes one <id>.out per case
to $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Standard-normal quantiles for the five requested levels.
Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Launch-night refresh stated in the rules: 638 -> 3,868 players.
REFRESH_PLAYERS_BEFORE = 638.0
REFRESH_PLAYERS_AFTER = 3868.0
REFRESH_FACTOR = REFRESH_PLAYERS_AFTER / REFRESH_PLAYERS_BEFORE

SIGMA_MIN = 0.15
SIGMA_MAX = 1.5
# Guard against a wild fit, not a prediction: launch week grows
# fast in both counts (the live page at ~2.25 days already shows
# Chapter 1 at 18% and the tracked set an order of magnitude past
# launch night), so the cap sits well above plausible growth.
GROWTH_CAP = 30.0
RW_SIGMA = 0.5


def parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) as aware UTC."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def u_of(dt):
    """Saturating time coordinate u = ln(1 + hours since launch)."""
    hours = (dt - LAUNCH).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 1e-6))


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line y ~ a + b*x.

    Returns None when there are fewer than two distinct x values.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def residual_std(xs, ys, line):
    """Population std of the residuals of y around the fitted line."""
    if line is None or len(xs) < 3:
        return 0.0
    intercept, slope = line
    sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(sq / len(xs))


def fit_log_rate(us, hs, ys):
    """Fit ln(y) against time and predict at the target.

    Fits against BOTH time coordinates -- u = ln(1 + hours since
    launch), which saturates, and raw hours, which lets growth run
    on -- and keeps the one with the smaller residual scatter, so
    the data's own shape picks the extrapolation instead of the
    builder guessing it. Returns (resid_std, line, kind) or None
    when no fit is possible.
    """
    line_u = least_squares(us, ys)
    line_h = least_squares(hs, ys)
    best = None
    if line_u is not None:
        best = (residual_std(us, ys, line_u), line_u, "u")
    if line_h is not None:
        r_h = residual_std(hs, ys, line_h)
        if best is None or r_h < best[0]:
            best = (r_h, line_h, "h")
    return best


def logit(pct):
    """Logit of a percent, clamped to [0.01, 99.99] first."""
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(eta):
    """Inverse logit, as a percent in [0.01, 100]."""
    if eta >= 0.0:
        p = 1.0 / (1.0 + math.exp(-eta))
    else:
        e = math.exp(eta)
        p = e / (1.0 + e)
    return min(max(p * 100.0, 0.01), 100.0)


def round2(x):
    """Round to 2 decimals (Exophase prints rates to 2 decimals)."""
    return math.floor(x * 100.0 + 0.5) / 100.0


def forecast_case(case):
    """Forecast one case dict; return the output dict for it."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_minutes, 0.0) / 1440.0

    target_dt = None
    if case.get("target"):
        target_dt = parse_time(case["target"])
    elif case.get("at"):
        target_dt = parse_time(case["at"])
    u_target = u_of(target_dt) if target_dt is not None else 0.0

    n_trophies = len(trophies)

    # ---- Collect the reads: (u, hours, players, [rates]) oldest first. ----
    reads = []
    for entry in history:
        try:
            at = parse_time(entry["at"])
        except (KeyError, TypeError, ValueError):
            continue
        players = entry.get("players_tracked")
        rates = entry.get("rates") or []
        if players is None or players <= 0:
            continue
        hours = (at - LAUNCH).total_seconds() / 3600.0
        reads.append((u_of(at), hours, float(players), rates))

    if not reads:
        # No history at all: fall back to a near-zero rate everywhere.
        return {
            "trophies": [
                {"id": t.get("id"), "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                for t in trophies
            ]
        }

    p_last = reads[-1][2]
    # Last non-null rate per trophy (for the flat / fallback paths).
    last_rate = []
    for i in range(n_trophies):
        rate = None
        for _, _, _, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(rate if rate is not None else 0.01)

    h_target = (
        (target_dt - LAUNCH).total_seconds() / 3600.0
        if target_dt is not None
        else 0.0
    )

    # ---- Players tracked at the target. ----
    p_values = [p for _, _, p, _ in reads]
    p_resid = 0.0
    if len(set(p_values)) > 1:
        best_p = fit_log_rate(
            [u for u, _, _, _ in reads],
            [h for _, h, _, _ in reads],
            [math.log(p) for _, _, p, _ in reads],
        )
        if best_p is not None:
            p_resid, line_p, kind_p = best_p
            x_target = u_target if kind_p == "u" else h_target
            p_target = math.exp(line_p[0] + line_p[1] * x_target)
        else:
            p_target = p_last
        # Never extrapolate wildly, and never forecast fewer players
        # than the last read saw (the tracked set only grows).
        p_target = min(max(p_target, p_last), p_last * GROWTH_CAP)
    else:
        # Flat history: Exophase has not refreshed yet. The next
        # refresh dilutes rates, and the rules state the launch-night
        # refresh outright (638 -> 3,868 players). Apply one such
        # refresh for a short horizon; for a long horizon let the
        # tracked set keep growing past it, mildly.
        if horizon_minutes <= 90.0:
            growth = REFRESH_FACTOR
        else:
            growth = REFRESH_FACTOR * (1.0 + horizon_days)
        growth = min(growth, GROWTH_CAP)
        p_target = p_last * growth

    rw = RW_SIGMA * math.sqrt(horizon_days)

    # ---- Per-trophy earner forecast. ----
    rows = []
    for i, trophy in enumerate(trophies):
        # Earner counts for the reads where this trophy has a rate.
        xs_u, xs_h, ys = [], [], []
        for u, hours, players, rates in reads:
            if i < len(rates) and rates[i] is not None:
                earners = int(round(float(rates[i]) * players / 100.0))
                xs_u.append(u)
                xs_h.append(hours)
                ys.append(math.log(max(earners, 1)))
        e_last = math.exp(ys[-1]) if xs_u else None
        flat_earners = len(set(ys)) <= 1 if ys else True

        if last_rate[i] >= 99.99:
            # The 100% trophy: every tracked player holds it, and the
            # first refresh left it at 100% while every other rate fell
            # (new players earn this one immediately). Do NOT dilute it:
            # its median stays at 99.99 and its band sits near 100.
            median = 99.99
            sigma = 0.5 if flat_earners else min(
                max(math.sqrt(rw * rw + p_resid * p_resid), SIGMA_MIN),
                SIGMA_MAX,
            )
        elif not xs_u:
            # Trophy never observed: near-zero median, wide band.
            median = 0.01
            sigma = 1.0
        elif flat_earners:
            # Earners do not move while the page is unrefreshed; the
            # rate falls purely because P grows.
            median = (math.exp(ys[0]) / p_target) * 100.0 if ys else 0.01
            # The refresh size is stated in the rules (638 -> 3,868),
            # and the first refresh left every rate but the 100% one
            # lower by dilution alone, so the centre is a real estimate
            # with small error: a tight band is the honest spread here.
            # A wide band would pay the spread cost on every item for
            # uncertainty the rules have already removed.
            sigma = 0.1
        else:
            best_e = fit_log_rate(xs_u, xs_h, ys)
            if best_e is None:
                e_target = math.exp(ys[-1])
                resid = 0.0
            else:
                resid, line, kind_e = best_e
                x_target = u_target if kind_e == "u" else h_target
                e_target = math.exp(line[0] + line[1] * x_target)
            # Earners cannot exceed the tracked players, and a fit
            # must not extrapolate wildly from the last observed count.
            # Earner counts are also monotone in reality (nobody
            # un-earns a trophy), so a fitted fall below the last
            # observed count is rounding noise: floor at E_last.
            if e_last is not None:
                e_target = min(e_target, e_last * GROWTH_CAP)
                e_target = max(e_target, e_last)
            e_target = min(max(e_target, 0.0), p_target)
            median = (e_target / p_target) * 100.0 if p_target > 0 else 0.01
            sigma = math.sqrt(rw * rw + resid * resid + p_resid * p_resid)
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)

        median = min(max(median, 0.01), 99.99)
        centre = logit(median)
        quantiles = []
        for z in Z_VALUES:
            quantiles.append(round2(inv_logit_pct(centre + z * sigma)))
        # Safety net: quantiles must be non-decreasing.
        quantiles.sort()
        rows.append({"id": trophy.get("id"), "q": quantiles})

    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
