# GBM Monte Carlo Option Pricer (C++17)

Monte Carlo pricing of European and Asian options under Geometric Brownian Motion,
benchmarked against closed-form Black–Scholes prices and Greeks.

## Features
- Exact GBM sampling for European payoffs: one normal draw per path, no discretisation error
- Streaming simulation: paths are never stored, so memory is O(threads) rather than O(paths)
- Antithetic variates for variance reduction
- Multithreaded, with an independent RNG stream per thread (splitmix64-derived seeds)
  and cache-line-aligned accumulators to avoid false sharing
- Greeks: pathwise delta and vega, likelihood-ratio gamma
- Scenario analysis: volatility sensitivity and moneyness
- Path-dependent example: arithmetic Asian call (252 time steps)

## Build and run
```
make
./gbm_mc [n_paths] [n_threads]      # defaults: 1,000,000 paths, all cores
```

## Method
Terminal price: S_T = S_0 · exp((r − σ²/2)T + σ√T·Z), Z ~ N(0,1).
Price = e^{−rT} · E[payoff]. Reported error bars are 95% confidence intervals.

| Greek | Estimator |
|-------|-----------|
| Delta | pathwise: e^{−rT} · 1{ITM} · S_T / S_0 |
| Vega  | pathwise: e^{−rT} · 1{ITM} · S_T (√T·Z − σT) |
| Gamma | likelihood ratio: e^{−rT} · payoff · ((Z²−1)/(S_0²σ²T) − Z/(S_0²σ√T)) |

## Results
Add your own measured output here (machine, thread count, timings).
