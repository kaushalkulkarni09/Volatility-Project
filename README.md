### **Hybrid Volatility Surface Modeling and Exotic Option Pricing**

This research presents a robust and comprehensive framework for volatility surface modeling and exotic option pricing. The project integrates three prominent financial models—the SSVI (Stochastic Volatility Inspired) model, the Local Volatility model, and the SABR (Stochastic Alpha Beta Rho) model—into a unified Python-based platform. By fetching real-time market data, the framework calibrates each model and employs them to price standard European options, calculate risk sensitivities (Greeks), and perform Monte Carlo simulations for complex, path-dependent exotic options, including Asian, Barrier, and Lookback options. The study provides a comparative analysis of the models' performance in fitting the market volatility smile and highlights the critical implications of model risk in the valuation of derivatives. 

**Introduction**

The valuation of financial derivatives, particularly exotic options, is fundamentally dependent on an accurate representation of the underlying asset's future price distribution. The Black-Scholes model, while foundational, assumes constant volatility, a premise contradicted by empirical observation. The market's volatility smile and skew reveal that implied volatility is a function of both strike price and time to maturity. This necessitates the use of more sophisticated models that can capture this dynamic structure, collectively known as the volatility surface.

This project addresses this challenge by implementing and comparing three state-of-the-art volatility surface models. Our goal is to create a practical, high-performance toolkit for a quantitative analyst to:

1. Accurately calibrate each model to real-world market data.

2. Assess their respective fits to the observed volatility surface.

3. Utilize the calibrated surfaces to price complex options and analyze their risk profiles.

The integration of these models into a single framework allows for a direct, side-by-side comparison of their performance, providing valuable insights into their practical application and limitations.

**Methodology and Model Implementation**

The framework is built in Python, leveraging key libraries for financial computation and data handling. The core components of the methodology are as follows:

**Data Acquisition and Pre-processing**

The system begins by connecting to the Yahoo Finance API via the yfinance library to fetch options chain data for a specified ticker (e.g., MSFT). The raw data, including market prices, strikes, and expiration dates, is then cleaned and processed to calculate time to maturity (T) and implied volatility (IV) for each option using a numerical solver for the Black-Scholes formula.

**Volatility Surface Models**

* **Smoothed SVI (SSVI):** his is a global model parameterized by a function of log-moneyness and time to maturity. It is defined by a set of six parameters [
rho,
eta,
gamma,
phi,
psi,p]. The calibration is an optimization problem minimized with respect to a penalized sum of squared errors, where penalties enforce static and calendar spread arbitrage-free conditions. The calibration is solved using the Differential Evolution algorithm, which is well-suited for high-dimensional, non-convex problems.

* **SABR Model:** The SABR model is a stochastic volatility model with an explicit, though approximate, formula for implied volatility. It is defined by four parameters: alpha (vol-of-vol), beta (elasticity), rho (correlation), and nu (volatility of volatility). This model is calibrated for each unique maturity slice, providing a precise fit to the volatility smile for a given expiration. The calibration is performed using a least-squares optimizer (scipy.optimize.least_squares) with strict bounds on the parameters.

* **Local Volatility Model:** The local volatility surface, 
sigma_LV(K,T), is derived from market-implied volatilities using Dupire’s formula. This is a two-step process:

1. For each maturity, a high-degree polynomial is fitted to the market's implied volatility smile to create a smooth, continuous function.

2. Dupire’s formula, a parabolic partial differential equation, is then applied to these polynomials to obtain the local volatility at any strike and time.

**Option Pricing and Greeks** 

The platform uses the calibrated models to perform key financial tasks. The Black-Scholes formula is adapted to use the implied volatility from each of the three models, allowing for direct pricing and comparison. Greeks (Delta, Gamma, Vega, Theta, Rho) are calculated using a finite difference approximation, providing a robust method for risk analysis that accounts for the volatility surface's curvature.

**Monte Carlo Simulation for Exotic Options**

For exotic options, an efficient Monte Carlo engine is implemented. The core innovation is that the instantaneous volatility used for each step of the simulated price path is not constant. Instead, it is dynamically retrieved from the chosen volatility surface model (LocalVol, SSVI, or SABR) at the current stock price and remaining time to maturity. This approach ensures the simulated paths are consistent with the market's observed volatility smile.

**Results and Discussion**

The following results were obtained from a live data run, demonstrating the framework's capabilities.

**Calibration Performance**

**SSVI:** The global SSVI calibration yielded a final SSE of 7.833, indicating a good overall fit across all strikes and maturities. The calibrated parameters were [0.0047, 0.1190, -0.0981, 0.9886, 1.0065, -0.0004].

**SABR:** The SABR calibration successfully fitted parameters for each maturity slice. The results showed reasonable parameter values, with beta often close to 0.99 (due to the boundary constraint), rho being negative (consistent with equity smiles), and nu varying across maturities.

**Local Volatility:** The polynomial fits for each maturity were highly accurate, with the derived local volatility surface showing the expected behavior: a higher volatility for out-of-the-money puts (low strikes) and a lower volatility for out-of-the-money calls (high strikes), consistent with the volatility skew.

**Option Pricing and Model Risk**

The comparison of model-derived prices for a sample option revealed significant discrepancies. For a call option with K=255 and T=0.3176, the market IV was 0.7036. The models, however, returned lower IVs: SSVI (0.4162), Local Vol (0.5807), and SABR (0.5642). This led to price differences of -$2.27 (SSVI), -$1.58 (Local Vol), and -$1.71 (SABR) relative to the market price.

These results are visually represented in the calibration plots.  The plots confirm that no single model perfectly captures every data point. The SABR and Local Vol models, calibrated per-maturity, provided a closer fit to the market IV for this specific option, demonstrating their strength in local interpolation. In contrast, the global SSVI model's objective is to provide a smooth, arbitrage-free fit to the entire surface, which may lead to a greater deviation for individual options.

| Model | Market Implied Volatility | Model Implied Volatility | Difference (IV) | Market Price | Model Price | Difference (Price) |
|---|---|---|---|---|---|---|
| SSVI | 0.7036 | 0.4162 | -0.2874 | 255.3500 | 253.0773 | -2.2727 |
| Local Vol | 0.7036 | 0.5807 | -0.1229 | 255.3500 | 253.7676 | -1.5824 |
| SABR | 0.7036 | 0.5642 | -0.1394 | 255.3500 | 253.6347 | -1.7153 |

**Exotic Option Valuation**

The Monte Carlo simulations highlighted the critical impact of model choice on exotic option pricing. The prices for the same Asian Call option varied by over 70% depending on the volatility model used ($13.48 for Local Vol vs. $22.94 for SSVI).

This discrepancy is a direct result of each model's implied risk-neutral distribution. The Local Volatility model correctly represents the dynamic, instantaneous volatility, making it theoretically the most consistent for path-dependent options. The SSVI and SABR models, as implied volatility surfaces, are more suited for pricing European options and may not fully capture the path-dependent nature of volatility, leading to varying results. This serves as a powerful illustration of model risk—the potential for financial losses due to the use of an inadequate or incorrect model.

| Option Type | Model | Price |
|---|---|---|
| Asian Call | Local Vol | 13.4814 |
| Asian Call | SSVI | 22.9380 |
| Asian Call | SABR | 15.9957 |
| Barrier Up-and-Out Call | Local Vol | 8.2182 |
| Barrier Up-and-In Call | SABR | 25.7500 |
| Lookback Floating Call | Local Vol | 42.6138 |
| Lookback Floating Put | SSVI | 99.3848 |
| Lookback Fixed Put | SABR | 48.4515 |

**Instructions for Reproducibility**

This project is designed for full reproducibility. All necessary dependencies and code structure are detailed below.

**System Requirements**

The following Python libraries are required for the simulation environment:

* yfinance
* numpy
* scipy
* pandas
* matplotlib

These dependencies can be installed via pip using the following command: **pip install yfinance numpy scipy pandas matplotlib**

**Execution**

The primary simulation script, main.py, will automatically download the necessary historical data, perform the comparative analysis, and generate the corresponding visualization plots. Execute the script from the terminal as follows: **python main.py**.

