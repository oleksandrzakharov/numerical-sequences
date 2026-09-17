# Is There Predictable Structure? A Workflow for Testing Numerical Sequences

The article and the report tool that goes with it.

| file | what |
|---|---|
| `article.pdf`, `article.md` | the article (typeset, and its Markdown source) |
| `fig1_sequences.png`, `fig2_pvalue_grid.png`, `fig3_calibration.png` | the article's three figures |
| `sequence_report.py` | one CSV in, one PDF report out: data checks, a reserved period, baselines, the diagnostics with their nulls, and, given a threshold-type event and costs, the probability check, the risk by time of the cycle, the scale-up / scale-down levels and the policy comparison on the reserved period, every section explained in plain words |
| `randomness_toolkit.py` | the diagnostics the report script uses (its only local dependency) |
| `sample_load.csv`, `sample_report.pdf` | the article's walkthrough series (a simulated server load factor, one value every 15 minutes) and the report the script produces from it |
| `requirements.txt` | Python packages |

## Running the report tool

```bash
pip install -r requirements.txt
python3 sequence_report.py sample_load.csv --time timestamp --value load_factor \
    --units "load factor" --threshold 0.86 --direction above --existing 0.80 \
    --cost-act 1 --cost-miss 20 --out sample_report.pdf
```

About two minutes for 17,280 rows. The PDF is rendered with a local Chrome or Chromium; without one the script writes the HTML and says so.

Any evenly sampled sequence works: `--value` names the column, `--time` an optional ISO-8601 timestamp column (delimiter and decimal comma are detected). Useful options: `--direction above|below|abs`, `--horizon` (steps ahead), `--transform diff|pct|logdiff` for a wandering quantity, `--period` to set or disable the cycle, `--cost-switch` to charge each change of state, `--reserve` for the share of data locked away. `python3 sequence_report.py -h` lists them all. Everything the report prints is reproducible from the CSV, the script and the settings listed at its end (seed 2025).

All data in this repository are simulated.
