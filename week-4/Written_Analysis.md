# Week 4 — Part A: Written Analysis
### ARIMA vs Prophet — Melbourne Daily Minimum Temperature (1981–1990)

**Results (last 20% of a 3-year subset held out as test, 219 days):**

| Model | MAE (°C) | RMSE (°C) | MAPE (%) |
|---|---|---|---|
| ARIMA(2,1,1) | 2.728 | 3.464 | 31.41 |
| **Prophet** | **1.979** | **2.568** | **21.95** |

Prophet outperformed ARIMA on every metric.

---

The gap is explained by the structure of the data rather than by either library being intrinsically
"better." Temperature has one dominant, highly regular annual cycle. Prophet is built around exactly
that shape: it explicitly decomposes the series into trend + yearly seasonality (+ optional holiday
terms) using an additive Generalized Additive Model, so it can represent the full year of seasonal
shape at once via Fourier terms, and it does not need the test window to be adjacent to a training
pattern it has already memorised locally.

ARIMA, by contrast, is a linear autoregressive model of *changes* in the series based on a handful of
recent lags — here, order `(2,1,1)` selected via `auto_arima` on AIC. With `seasonal=False`, this ARIMA
has no explicit mechanism for a 365-day cycle; it can only approximate seasonality indirectly through
autocorrelation in nearby lags, which decays quickly. As the forecast horizon extends across the
219-day test window, an ARIMA forecast without a seasonal component tends to flatten toward its
long-run mean, while Prophet continues to trace the expected seasonal curve. This is visible when
comparing the two forecast-vs-actual plots in the notebook (Sections A3.1 and A4): Prophet's curve
continues to rise and fall with the season; ARIMA's forecast is comparatively flat and drifts from the
true seasonal trajectory the further out it predicts.

The ADF test on the full series (p ≈ 0.00025) already confirmed the data is stationary in levels —
unsurprising for a physically bounded, mean-reverting quantity like temperature, unlike e.g. stock
prices. That stationarity is exactly why ARIMA's simple differencing captures short-run dynamics
reasonably well (d=1 was sufficient) but still cannot compensate for the missing seasonal term.

A fairer ARIMA comparison would use a seasonal SARIMA with `m=365`, which is far more expensive to fit
but would let ARIMA also model the annual cycle explicitly — a natural next experiment worth running if
compute time allows. For this dataset's use case (long-horizon seasonal forecasting), Prophet's built-in
seasonality handling makes it the more practical default; ARIMA/SARIMA remains preferable for
shorter-horizon series with irregular or non-seasonal autocorrelation structure, or when a fully
interpretable linear model with well-understood statistical properties (confidence intervals, hypothesis
tests on coefficients) is required for the downstream use case.

*(≈300 words)*
