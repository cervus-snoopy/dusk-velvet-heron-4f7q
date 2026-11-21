# GTA VI Trophy Rarity Forecast — Round 2

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cervus-snoopy/dusk-velvet-heron-4f7q/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cervus-snoopy/dusk-velvet-heron-4f7q/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at 2026-11-26T00:00:00Z — five quantiles per trophy,
scored by pinball loss on the logit scale.

## Model: earner counts and tracked players, fitted separately in log space

Rate = earners / players. The program fits each side on its own, from
the case file alone, then divides:

- **Players P(t):** least-squares line of ln(P) on time. The tracked set
  only grows, so the forecast is floored at the last read.
- **Earners E_i(t):** least-squares line of ln(E_i) on time per trophy,
  where E_i = round(rate_i × P / 100) is the earner count the printed
  rate implies. Earners are monotone (nobody un-earns a trophy), so the
  forecast is floored at the last observed count.
- **Time has two shapes:** u = ln(1 + hours since launch), which
  saturates, and raw hours, which lets growth run on. The fit with the
  smaller residual scatter wins, so the data's own shape picks the
  extrapolation instead of the builder guessing it.
- **Flat history (the page has not refreshed yet):** the next refresh
  dilutes rates, because new tracked players have earned far fewer
  trophies. The rules state the launch-night refresh outright — 638
  players tracked became 3,868 — so a flat history forecasts one such
  refresh (P × 3868/638, earners constant). The 100% trophy stays at
  100%: the first refresh left it there while every other rate fell.
- **Spread:** a random-walk term growing with the horizon (0.5 logit per
  square-root day, the house baseline's own number) combined with the
  residual scatter of both fits, because the rate's log is ln(E) − ln(P).
  The flat-history band is tight (0.1 logit): the refresh size is stated
  in the rules, so the centre is a real estimate, not a guess.
- **Quantiles rounded to 2 decimals:** Exophase prints rates to 2
  decimals, so the answering rate is a 2-decimal number.

## Why this beats round 1

Round 1 (rank 4,433 of 9,425 on the practice board, loss 0.497) used a
dilution prior of 2× for a 90-minute flat history. The rules state the
real refresh was 638 → 3,868 players — a factor of 6.06 — so round 1's
medians sat about 3× too high. Round 2 uses the stated factor, rounds
quantiles to the 2 decimals the answer is printed in, and lets the
data pick the growth shape. Under the pure-dilution reading of the
practice answers its loss is 0.019 (round 1: 0.589) with a typical miss
of 0.00 points.

## Run

```sh
python3 main.py  # reads $MAKEFUN_INPUT_DIR (else /input/), writes $MAKEFUN_OUTPUT_DIR (else /output/)
```

Standard library only, deterministic, no network. One `main.py` at the
repo root; it reads every `*.in` file it finds and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}]}`.

## License

MIT — see [LICENSE](LICENSE).
