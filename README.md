# BNBL Stock Forecasting — Reproduction Code

Source code, data and reproduction scripts for the manuscript
"Comparative Analysis of Statistical, Machine Learning and Deep Learning
Models for Forecasting Bhutan National Bank Limited (BNBL) Stock Prices."

## Repository contents

```
.
├── BNBL_price_report_All.csv          # raw daily closing-price file (see note below)
├── Arima & Holt winter.ipynb          # ARIMA + Holt-Winters; multi-window backtest;
│                                      # Ljung-Box residual diagnostics; Holt-Winters
│                                      # AIC comparison
├── Machine learning models.ipynb      # LR, SVR, RF, XGBoost, LightGBM; multi-window;
│                                      # ablation; Diebold-Mariano; multi-seed;
│                                      # naive lag-1 baseline
├── Deep Learning Models.ipynb         # LSTM, GRU, CNN, CNN-LSTM with KerasTuner;
│                                      # multi-window cell
├── scripts/                           # standalone reproductions
│   ├── multiwindow_backtest.py        # K=6 expanding-window stat+ML run
│   ├── multiwindow_dl.py              # K=6 backtest for DL with fixed archs
│   ├── ablation.py                    # leave-one-feature-group-out
│   ├── significance_test.py           # pairwise DM with HLN + Holm
│   └── seed_robustness.py             # 5 ML seeds + 3 DL seeds
├── environment.yml                    # conda environment specification
└── README.md
```

### A note on the source data

`BNBL_price_report_All.csv` is the raw closing-price report as supplied by
the Royal Securities Exchange of Bhutan, with one row per intraday tick. The
file therefore contains multiple rows for the same trading day; every notebook
and script aggregates to one row per day via
`df.groupby(df['Date'].dt.date)['Close Price'].last()` before any modelling.
No information is dropped — only the within-day duplicates are collapsed
to the day's closing price.

## Viewing the executed notebooks

The notebooks in this repository are stored with their cell outputs
intact (so reviewers can see the results without running anything).
The Machine-learning-models notebook is approximately 24 MB, which
exceeds the size at which GitHub's own notebook viewer falls back to a
raw-JSON view. Use [nbviewer](https://nbviewer.org), which is run by
the Jupyter project and renders any size, to read the executed
notebooks in a browser:

- **Statistical**: <https://nbviewer.org/github/uggxy/Bht-Stockexchange/blob/main/Arima%20%26%20Holt%20winter.ipynb>
- **Machine learning**: <https://nbviewer.org/github/uggxy/Bht-Stockexchange/blob/main/Machine%20learning%20models.ipynb>
- **Deep learning**: <https://nbviewer.org/github/uggxy/Bht-Stockexchange/blob/main/Deep%20Learning%20Models.ipynb>

## Reproducing the analysis

1. **Clone and create the environment**
   ```bash
   git clone https://github.com/uggxy/Bht-Stockexchange.git
   cd Bht-Stockexchange
   conda env create -f environment.yml
   conda activate stock-forecast
   ```

2. **Run the notebooks in order** (in the JupyterLab/VS Code kernel
   `stock-forecast`):
   - `Statistical model/Arima & Holt winter.ipynb` — produces ARIMA and
     Holt-Winters one-step-ahead forecasts and the multi-window backtest.
   - `Machine learning model/Machine learning models.ipynb` — produces all
     ML model results, the K=6 multi-window backtest (Step 9), the
     feature-group ablation (Step 10), the Diebold–Mariano significance
     test (Step 11), and the multi-seed robustness check (Step 12).
   - `Deep Learning Models/Deep Learning Models.ipynb` — KerasTuner-tuned
     single-window run plus the K=6 multi-window backtest (Step 14).

3. **Or run the standalone scripts** to reproduce each table directly:
   ```bash
   python scripts/multiwindow_backtest.py     # Table 1 stat+ML rows
   python scripts/multiwindow_dl.py           # Table 1 DL rows
   python scripts/ablation.py                 # Table 2
   python scripts/significance_test.py        # Tables 3 + DM matrices
   python scripts/seed_robustness.py          # Table 4
   ```
   Each script writes a CSV under `/tmp/` (or the working directory) with
   the exact numbers reported in the paper.

## Random seeds

All stochastic models are seeded for reproducibility:

| Model class           | Seed(s)              |
|-----------------------|----------------------|
| Random Forest         | 42 (paper), 0–4 (multi-seed robustness) |
| XGBoost, LightGBM     | 42 (paper), 0–4 (multi-seed robustness) |
| LSTM, GRU, CNN, CNN-LSTM | 42 (paper), 0–2 (multi-seed robustness) |

Linear Regression, SVR, ARIMA and Holt-Winters are deterministic given
the input data.

## Backtest protocol

- **Hold-out**: last 30 trading days of the series.
- **Multi-window**: K = 6 non-overlapping expanding-window test slices of
  H = 30 days each, walking backward from the most recent date. Total
  pooled test set: 180 forecasts per model.
- **One-step-ahead rolling forecast** within each window: at each step t
  the statistical models are refit on observations up to t − 1 and
  forecast a single step; the ML and DL models predict t from features
  derived from observations up to t − 1.
- **Feature engineering** (leak-fixed): seven lag features, rolling means
  and standard deviations over 3-, 5-, 10- and 20-day windows on the
  closing price shifted by one day, daily return and 5-day rolling
  volatility lagged by one period to prevent target leakage.

## Citation

If you use this code or data, please cite:

> [Author], (2026). Comparative Analysis of Statistical, Machine Learning
> and Deep Learning Models for Forecasting Bhutan National Bank Limited
> (BNBL) Stock Prices. [Journal], [vol/issue], [pages]. DOI: [to appear]

## License

[Add a license — MIT and Apache-2.0 are common choices for research
code; CC-BY-4.0 is common for the dataset if licensing requires it.]
