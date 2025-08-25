# volatility-project.py

import numpy as np
from scipy.stats import norm
import scipy.optimize as sco
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime
import json
import warnings
from typing import Union, Optional 


warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.shared")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in double_scalars") # For very small sigma in BS
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log") # For K near 0 or S near 0


# --- Configuration and Global Variables ---
# Application ID for potential Firebase integration (not directly used in this script's core logic)
appId = "hybrid-vol-model-app"
try:
    if '__app_id' in globals():
        appId = __app_id
except NameError:
    pass

# Firebase configuration (similarly, for potential external integration)
firebaseConfig = {}
try:
    if '__firebase_config' in globals():
        firebaseConfig = json.loads(__firebase_config)
except NameError:
    print("Warning: __firebase_config is not defined. Using empty config.")
    firebaseConfig = {}
except json.JSONDecodeError:
    print("Warning: __firebase_config is not valid JSON. Using empty config.")
    firebaseConfig = {}


class HybridVolatilitySurfaceModel:
    """
    A comprehensive hybrid class to implement the SSVI (Smoothed SVI),
    Local Volatility Surface, and SABR (Stochastic Alpha, Beta, Rho) parameterization models.

    The model provides a unified framework for analyzing, calibrating, and
    deriving risk metrics from implied volatility surfaces using three distinct
    yet powerful approaches, now also including Monte Carlo simulation for exotic options:

    1.  SSVI Model: A parametric, arbitrage-free (under conditions) 3D surface
        calibrated globally to market data.
    2.  Local Volatility Model: A non-parametric surface derived via Dupire's formula,
        which perfectly fits observed option prices (implied volatilities).
    3.  SABR Model: A popular parametric model for single maturity smiles,
        calibrated per-maturity slice.
    4.  Monte Carlo Simulation: For pricing path-dependent exotic options like Asian,
        Barrier, and Lookback options, leveraging the derived volatility surfaces
        for realistic price path generation.

    Features include:
    - Robust real-world option data fetching from yfinance.
    - Black-Scholes option pricing and implied volatility calculation.
    - SSVI total variance parameterization with time-dependent components and global calibration.
    - Local Volatility surface derivation using Dupire's formula,
      based on polynomial fitting of implied volatility smiles.
    - SABR implied volatility approximation and per-maturity calibration.
    - Incorporation of static and calendar spread arbitrage-free penalties during SSVI calibration.
    - Comprehensive calculation of option Greeks (Delta, Gamma, Vega, Theta, Rho)
      via finite differences, applicable to either the SSVI, Local Volatility, or SABR model.
    - Advanced visualizations for all calibrated surfaces (smiles per maturity)
      and detailed Greek profiles across strikes and maturities.
    - Monte Carlo simulation for pricing Arithmetic Mean Asian Options (fixed strike).
    - Monte Carlo simulation for pricing Barrier Options (e.g., Up-and-Out, Down-and-In).
    - Monte Carlo simulation for pricing Lookback Options (fixed and floating strike).
    """

    def __init__(self, S0: float, r: float, q: float):
        """
        Initializes the HybridVolatilitySurfaceModel with current market conditions.

        Args:
            S0 (float): Current underlying asset price. This will be updated by fetched data.
            r (float): Risk-free interest rate (annualized, as a decimal).
            q (float): Dividend yield (annualized, as a decimal).
        """
        self.S0 = S0
        self.r = r
        self.q = q

        # DataFrame to store all fetched market option data, including implied volatilities.
        self.market_option_data: Optional[pd.DataFrame] = None
        # Ticker symbol of the underlying asset (e.g., 'MSFT').
        self.ticker_symbol: Optional[str] = None
        # List of tuples storing fetched expiration dates and their corresponding
        # time to maturities in years: [(date_str, TTM_years)].
        self.fetched_expirations: list[tuple[str, float]] = []

        # --- SSVI Model Parameters ---
        # Global SSVI parameters for the entire surface:
        # [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power]
        # theta_T = alpha0 + alpha1*T + alpha2*T^2 (ATM total variance)
        # psi_T = eta / (theta_T^gamma_power) (Curvature related parameter)
        # rho_bar = constant correlation parameter
        self.global_ssvi_params: Optional[list[float]] = None

        # --- Local Volatility Model Parameters ---
        # Dictionary to store polynomial coefficients for implied volatility per maturity.
        # Format: {T_val: [coeff_0, coeff_1, coeff_2, ...]} (for fitting sigma_implied(k))
        self.poly_coeffs_implied_vol: dict[float, np.ndarray] = {}
        # Dictionary to store the derived local volatility value at a (K, T) grid.
        # This will be populated after Dupire's formula is applied.
        self.local_vol_surface: dict[float, dict[float, float]] = {}

        # --- SABR Model Parameters ---
        # Dictionary to store calibrated SABR parameters for each maturity slice.
        # Format: {T_val: [alpha, beta, rho, nu]}
        self.sabr_params_per_maturity: dict[float, list[float]] = {}


    def fetch_stock_data(self, ticker_symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical stock data from yfinance to get the latest spot price (S0).
        Automatically adjusts the initial S0 to the latest closing price.

        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., 'MSFT', 'AAPL').
            start_date (str, optional): Start date for historical data in 'YYYY-MM-DD' format.
                                        Defaults to 2 years prior to today if None.
            end_date (str, optional): End date for historical data in 'YYYY-MM-DD' format.
                                      Defaults to today if None.

        Returns:
            pandas.DataFrame: A DataFrame containing historical stock data.
                              Returns None if data fetching fails or no data is found.
        """
        print(f"Fetching historical data for {ticker_symbol} from yfinance...")

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.date.today().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')

        try:
            # Download stock data with auto_adjust=True to get adjusted close prices
            data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                print(f"No historical data found for {ticker_symbol} in the specified date range [{start_date} to {end_date}].")
                return None

            # Update the model's current spot price to the latest closing price
            self.S0 = float(data['Close'].iloc[-1])
            print(f"Updated S0 to latest close price: {self.S0:.2f} for {ticker_symbol}")
            return data
        except Exception as e:
            print(f"Error fetching historical stock data for {ticker_symbol}: {e}")
            return None

    def fetch_option_data(self, ticker_symbol: str, num_expirations: int = 10) -> Optional[pd.DataFrame]:
        """
        Fetches option chain data for multiple maturities from yfinance.
        It prioritizes maturities at least 90 days out and then selects the
        'num_expirations' closest suitable ones to ensure sufficient time to maturity
        for more stable implied volatility calculations and model fitting.

        Args:
            ticker_symbol (str): The stock ticker symbol.
            num_expirations (int, optional): The desired number of expiration dates to fetch.
                                              Defaults to 3.

        Returns:
            pandas.DataFrame: A consolidated DataFrame of call and put options
                              for the selected maturities. Returns None if fetching fails
                              or no valid options are found after filtering.
        """
        print(f"\nFetching option chains for {ticker_symbol} for multiple maturities...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            all_expirations = ticker.options # Get all available expiration dates from yfinance
            if not all_expirations:
                print(f"No expiration dates found for {ticker_symbol}.")
                return None

            today = datetime.date.today()
            suitable_expirations = []

            # Filter for expirations with at least 90 days to maturity for better model stability
            for exp_str in all_expirations:
                exp_date = datetime.datetime.strptime(exp_str, '%Y-%m-%d').date()
                ttm_days = (exp_date - today).days
                if ttm_days >= 90: # Minimum TTM in days
                    suitable_expirations.append((exp_str, ttm_days))

            # Sort the suitable expirations by their time to maturity (ascending order)
            suitable_expirations.sort(key=lambda x: x[1])
            # Select the top 'num_expirations' closest maturities
            selected_exp_strs = [exp_str for exp_str, _ in suitable_expirations[:num_expirations]]

            if not selected_exp_strs:
                print(f"No suitable expiration dates (>=90 days) found. Using first available {num_expirations} dates as a fallback.")
                selected_exp_strs = all_expirations[:num_expirations] # Fallback to any available if none meet criteria


            all_option_data = pd.DataFrame() # Initialize an empty DataFrame to store all option data
            self.fetched_expirations = [] # Reset the list of fetched expirations for a fresh run

            for exp_str in selected_exp_strs:
                try:
                    # Fetch option chain for the current expiration date
                    option_chain = ticker.option_chain(exp_str)

                    # Extract relevant columns for calls and puts
                    calls = option_chain.calls[['strike', 'lastPrice', 'bid', 'ask']].copy()
                    calls['option_type'] = 'call'
                    puts = option_chain.puts[['strike', 'lastPrice', 'bid', 'ask']].copy()
                    puts['option_type'] = 'put'

                    # Calculate time to maturity (T) in years for the current expiration
                    current_ttm_days = (datetime.datetime.strptime(exp_str, '%Y-%m-%d').date() - today).days
                    current_T = current_ttm_days / 365.25 # Convert days to years

                    # Add T and expiration date columns to both call and put DataFrames
                    calls['T'] = current_T
                    puts['T'] = current_T
                    calls['expiration_date'] = exp_str
                    puts['expiration_date'] = exp_str

                    # Calculate mid-price (average of bid and ask) as the market price
                    # Fill NaN mid-prices with lastPrice (if bid/ask are missing)
                    calls['market_price'] = (calls['bid'] + calls['ask']) / 2
                    calls['market_price'] = calls['market_price'].fillna(calls['lastPrice'])
                    puts['market_price'] = (puts['bid'] + puts['ask']) / 2
                    puts['market_price'] = puts['market_price'].fillna(puts['lastPrice'])

                    # --- Robust Filtering of Illiquid/Noisy Options ---
                    # 1. Ensure positive bid and ask prices (filters out options not actively quoted)
                    # 2. Filter by reasonable bid-ask spread to avoid illiquid options (e.g., spread < 50% of mid-price)
                    calls_filtered = calls[(calls['bid'] > 0) & (calls['ask'] > 0) & ((calls['ask'] - calls['bid']) / calls['market_price'] < 0.5)]
                    puts_filtered = puts[(puts['bid'] > 0) & (puts['ask'] > 0) & ((puts['ask'] - puts['bid']) / puts['market_price'] < 0.5)]

                    # 3. Filter for options with strike within a certain percentage of S0 (e.g., +/- 25%)
                    # This removes very far OTM/ITM options which often have unreliable implied volatilities.
                    strike_filter_percent = 0.5
                    calls_filtered = calls_filtered[np.abs(calls_filtered['strike'] - self.S0) / self.S0 < strike_filter_percent]
                    puts_filtered = puts_filtered[np.abs(puts_filtered['strike'] - self.S0) / self.S0 < strike_filter_percent]

                    # Concatenate filtered calls and puts to the overall DataFrame
                    if not calls_filtered.empty:
                        all_option_data = pd.concat([all_option_data, calls_filtered], ignore_index=True)
                    if not puts_filtered.empty:
                        all_option_data = pd.concat([all_option_data, puts_filtered], ignore_index=True)

                    # Add to fetched_expirations only if valid options were found for this maturity
                    if not (calls_filtered.empty and puts_filtered.empty):
                        self.fetched_expirations.append((exp_str, current_T))
                        print(f"  Fetched {len(calls_filtered) + len(puts_filtered)} valid options for expiration {exp_str} (T={current_T:.4f} years).")
                except Exception as inner_e:
                    print(f"  Error fetching data for expiration {exp_str}: {inner_e}")
                    continue

            self.market_option_data = all_option_data # Store the combined market data

            if self.market_option_data.empty:
                print(f"No valid option data found for {ticker_symbol} across selected expirations after filtering.")
                return None

            print(f"Total fetched {len(self.market_option_data)} valid options across {len(self.fetched_expirations)} maturities.")
            return self.market_option_data

        except Exception as e:
            print(f"Error fetching option data for {ticker_symbol}: {e}")
            self.market_option_data = None
            self.fetched_expirations = []
            return None

    def black_scholes_price(self, S: float, K: float, T: float, r: float, q: float, sigma_val: float, option_type: str = 'call') -> float:
        """
        Calculates the Black-Scholes option price (call or put).
        This function is independent of the class instance's S0, r, q, allowing for
        flexibility in Greek calculations where these parameters are perturbed.

        Args:
            S (float): Underlying asset price.
            K (float): Strike price.
            T (float): Time to maturity (in years).
            r (float): Risk-free rate (annualized).
            q (float): Dividend yield (annualized).
            sigma_val (float): Volatility (annualized, as a decimal).
            option_type (str, optional): Type of option: 'call' or 'put'. Defaults to 'call'.

        Returns:
            float: The Black-Scholes option price. Returns 0.0 or intrinsic value for edge cases.
        """
        # Handle cases where T or sigma are extremely small to prevent division by zero or log(0)
        if T <= 1e-10 or sigma_val <= 1e-10:
            if option_type == 'call':
                return np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            elif option_type == 'put':
                # Corrected from S*exp(-q*T) to S*exp(-r*T) for consistency with K*exp(-r*T) in intrinsic value calc.
                return np.maximum(K * np.exp(-r * T) - S * np.exp(-r * T), 0)
            else:
                return 0.0

        # Ensure sigma_val is positive to avoid math domain errors
        sigma_val = np.maximum(sigma_val, 1e-10)

        # Calculate d1 and d2 terms of the Black-Scholes formula
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T) / (sigma_val * np.sqrt(T))
        d2 = d1 - sigma_val * np.sqrt(T)

        # Calculate option price based on type
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Clip the price to ensure it's non-negative and not excessively large (e.g., max S * 2)
        return np.clip(price, 0.0, S * 2)

    def implied_volatility(self, market_price: float, strike: float, T_val: float, option_type: str = 'call',
                           S_override: Optional[float] = None, r_override: Optional[float] = None, q_override: Optional[float] = None) -> float:
        """
        Calculates the Black-Scholes implied volatility (IV) for a given market option price.
        It uses a root-finding algorithm (scipy.optimize.brentq) to invert the Black-Scholes formula.
        Overrides for S, r, q are provided for use in Greek calculations where these parameters are perturbed.

        Args:
            market_price (float): The observed market price of the option.
            strike (float): The strike price of the option.
            T_val (float): Time to maturity of the option (in years).
            option_type (str, optional): Type of option: 'call' or 'put'. Defaults to 'call'.
            S_override (float, optional): Optional override for the underlying asset price.
                                          Uses `self.S0` if None.
            r_override (float, optional): Optional override for the risk-free rate.
                                          Uses `self.r` if None.
            q_override (float, optional): Optional override for the dividend yield.
                                          Uses `self.q` if None.

        Returns:
            float: The calculated implied volatility. Returns `np.nan` if a valid IV
                   cannot be found.
        """
        if market_price <= 0:
            return np.nan # Implied volatility is not defined for zero or negative prices

        # Use overrides if provided, otherwise fall back to model's current parameters
        S_current = S_override if S_override is not None else self.S0
        r_current = r_override if r_override is not None else self.r
        q_current = q_override if q_override is not None else self.q

        def f(sigma_candidate: float) -> float:
            """
            Inner function for `brentq` to find the root.
            Calculates the difference between the Black-Scholes price (for a given sigma)
            and the target market price (or its equivalent for puts).
            """
            # For puts, convert market price to an equivalent call price using Put-Call Parity
            if option_type == 'put':
                call_price_equivalent = market_price + S_current * np.exp(-q_current * T_val) - strike * np.exp(-r_current * T_val)
                # Ensure the equivalent call price is non-negative to avoid issues in BS formula
                call_price_equivalent = np.maximum(call_price_equivalent, 1e-6)
                return self.black_scholes_price(S_current, strike, T_val, r_current, q_current, sigma_candidate, option_type='call') - call_price_equivalent
            else: # 'call'
                return self.black_scholes_price(S_current, strike, T_val, r_current, q_current, sigma_candidate, option_type='call') - market_price

        # Define search bounds for volatility: very small positive to 500%
        # These bounds are generally sufficient for most equity options.
        a, b = 1e-8, 5.0

        # --- Static Arbitrage Checks (filters out options that cannot be priced) ---
        # If market price violates basic no-arbitrage bounds, implied volatility is undefined.
        # Add a small tolerance (e.g., 1e-4) for floating-point comparisons.
        intrinsic_value_call = np.maximum(S_current * np.exp(-q_current * T_val) - strike * np.exp(-r_current * T_val), 0)
        intrinsic_value_put = np.maximum(strike * np.exp(-r_current * T_val) - S_current * np.exp(-r_current * T_val), 0)

        # Check for market price below intrinsic value (arbitrage for European options)
        if option_type == 'call' and market_price < intrinsic_value_call - 1e-4:
            return np.nan
        if option_type == 'put' and market_price < intrinsic_value_put - 1e-4:
            return np.nan
        # Check for market price above theoretical max (for calls)
        if option_type == 'call' and market_price > S_current * np.exp(-q_current * T_val) + 1e-4:
            return np.nan
        # Check for market price above theoretical max (for puts)
        if option_type == 'put' and market_price > strike * np.exp(-r_current * T_val) + 1e-4:
            return np.nan

        try:
            # `brentq` requires that f(a) and f(b) have opposite signs (i.e., a root is bracketed)
            if f(a) * f(b) > 0:
                return np.nan # No root found within the given bounds

            # Use Brent's method to find the root (implied volatility)
            iv = sco.brentq(f, a, b, xtol=1e-6) # xtol: absolute tolerance for the root
            return iv
        except ValueError: # Occurs if brentq fails (e.g., due to numerical issues or non-monotonicity of f)
            return np.nan
        except Exception as e:
            # Catch any other unexpected errors during IV calculation
            # print(f"Error computing IV for K={strike}, T={T_val}, Price={market_price}, Type={option_type}: {e}")
            return np.nan

    # --- SSVI Model Specific Methods ---
    def get_ssvi_params_at_T(self, T_val: float, global_ssvi_params: list[float]) -> dict[str, float]:
        """
        Calculates the time-dependent SSVI parameters (theta_T, psi_T, rho_bar)
        from the global SSVI parameters for a given time to maturity T.

        Args:
            T_val (float): Time to maturity in years.
            global_ssvi_params (list): The global SSVI parameters:
                                       [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power].

        Returns:
            dict: A dictionary containing 'theta_T', 'psi_T', and 'rho_bar' for the given T.
        """
        alpha0, alpha1, alpha2, rho_bar, eta, gamma_power = global_ssvi_params

        # Ensure T_val is positive for calculations involving powers of T
        T_val = np.maximum(T_val, 1e-10)

        # Calculate theta_T (ATM total variance) as a quadratic function of T
        theta_T = alpha0 + alpha1 * T_val + alpha2 * (T_val**2)
        theta_T = np.maximum(theta_T, 1e-10) # Ensure theta_T is non-negative

        # Calculate psi_T (curvature related parameter) as a power law of theta_T
        # Handle cases where theta_T is very small or zero to avoid division by zero or large numbers
        if theta_T < 1e-8:
            psi_T = eta / (1e-8**gamma_power) # Use a tiny value to prevent inf
        else:
            psi_T = eta / (theta_T**gamma_power)

        # Ensure psi_T is within reasonable bounds
        psi_T = np.clip(psi_T, 1e-8, 10.0)

        # rho_bar is a constant global parameter
        rho_bar = np.clip(rho_bar, -0.9999, 0.9999) # Ensure rho_bar is within [-1, 1]

        return {
            'theta_T': theta_T,
            'psi_T': psi_T,
            'rho_bar': rho_bar
        }

    def ssvi_total_variance(self, log_moneyness: float, T_val: float, global_ssvi_params: list[float]) -> float:
        """
        Calculates the SSVI total implied variance (w = sigma^2 * T) for a given
        log-moneyness (k) and time to maturity (T), using the global SSVI parameters.
        The SSVI formulation used here is based on Gatheral's framework, which dynamically
        links the SVI parameters across maturities to form a smooth surface.

        Formula: w(k, T) = theta_T / 2 * [1 - rho_bar * psi_T * k + sqrt((psi_T * k - rho_bar)^2 + (1 - rho_bar^2))]

        Args:
            log_moneyness (float): k = ln(K/S0) for simplicity. In a formal context, k = ln(K/F_T).
            T_val (float): Time to maturity in years.
            global_ssvi_params (list): The global SSVI parameters:
                                       [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power].

        Returns:
            float: The SSVI total implied variance (w). Ensures w is non-negative.
        """
        # Get the time-dependent SSVI parameters (theta_T, psi_T, rho_bar)
        # These are derived from the global parameters and the current T_val
        derived_params = self.get_ssvi_params_at_T(T_val, global_ssvi_params)
        theta_T = derived_params['theta_T']
        psi_T = derived_params['psi_T']
        rho_bar = derived_params['rho_bar']

        # Ensure parameters are valid before calculation
        if theta_T <= 1e-10:
            return 1e-10 # Return a small positive value if ATM variance is zero or negative

        # Calculate the square root term. np.maximum ensures argument is non-negative.
        term_sqrt_arg = (psi_T * log_moneyness - rho_bar)**2 + (1 - rho_bar**2)
        term_sqrt = np.sqrt(np.maximum(term_sqrt_arg, 1e-10))

        # Calculate the SSVI total variance
        w = (theta_T / 2.0) * (1 - rho_bar * psi_T * log_moneyness + term_sqrt)

        # Ensure total variance is always non-negative
        return np.maximum(w, 1e-10)


    def ssvi_implied_volatility(self, strike: float, T_val: float, global_ssvi_params: list[float], S0_override: Optional[float] = None) -> float:
        """
        Calculates the SSVI implied volatility (annualized standard deviation)
        for a given strike and time to maturity, using the global SSVI parameters.
        Includes S0_override for Monte Carlo/Greek calculations.

        Args:
            strike (float): Strike price.
            T_val (float): Time to maturity (in years).
            global_ssvi_params (list): The global SSVI parameters:
                                       [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power].
            S0_override (float, optional): Override for the underlying asset price when calculating moneyness.
                                           Used in Monte Carlo to get vol at current path S.

        Returns:
            float: The annualized SSVI implied volatility. Returns `np.nan` for very short maturities.
        """
        if T_val <= 1e-10: # Implied volatility is undefined for zero or near-zero time to maturity
            return np.nan

        current_S_for_moneyness = S0_override if S0_override is not None else self.S0
        log_moneyness = np.log(np.maximum(strike / current_S_for_moneyness, 1e-10))

        # Calculate the total implied variance from the SSVI function
        total_variance = self.ssvi_total_variance(log_moneyness, T_val, global_ssvi_params)

        # Implied volatility is the square root of total variance divided by time to maturity
        iv = np.sqrt(total_variance / T_val)

        # Clip the implied volatility to reasonable bounds (e.g., 0.1% to 500%)
        # This prevents extreme values that could cause numerical issues in Black-Scholes.
        return np.clip(iv, 1e-8, 5.0)

    def _objective_function_ssvi(self, global_ssvi_calib_params: list[float],
                                  market_strikes: np.ndarray, market_TTMs: np.ndarray,
                                  market_implied_vols: np.ndarray) -> float:
        """
        The global objective function to minimize during SSVI calibration.
        Calculates the sum of squared errors (SSE) between market implied volatilities
        and SSVI model-predicted implied volatilities across ALL options and maturities.
        Includes penalties for static and calendar spread arbitrage conditions.

        Args:
            global_ssvi_calib_params (list): The 6 global SSVI parameters being optimized:
                                             [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power].
            market_strikes (np.ndarray): Array of all strike prices from market data.
            market_TTMs (np.ndarray): Array of all time to maturities from market data.
            market_implied_vols (np.ndarray): Array of all market implied volatilities.

        Returns:
            float: The total objective value (SSE + penalty), which is minimized.
        """
        alpha0, alpha1, alpha2, rho_bar, eta, gamma_power = global_ssvi_calib_params

        # --- Parameter Validity and Arbitrage-Free Penalties ---
        penalty = 0.0

        # 1. Ensure parameters related to level (alpha0, eta) are positive
        if alpha0 < 0 or eta < 0:
            penalty += 1e10 # Huge penalty if base variance or curvature scale is negative

        # 2. Ensure rho_bar is within [-1, 1]
        if rho_bar < -0.9999 or rho_bar > 0.9999: # Small buffer from -1 and 1
            penalty += 1e10

        # 3. Penalize unreasonable magnitudes for other parameters
        if np.abs(alpha1) > 5.0 or np.abs(alpha2) > 5.0 or np.abs(gamma_power) > 5.0:
            penalty += 1e8

        # 4. Calendar Spread Arbitrage Condition: theta_T must be non-decreasing with T.
        # This is a crucial condition for a 3D arbitrage-free surface.
        # Check this for a few representative T values or a dense range.
        # Use a numerical derivative approximation.
        ttm_sorted_unique = sorted(list(set(market_TTMs))) # Get unique sorted maturities
        if len(ttm_sorted_unique) > 1:
            for i in range(len(ttm_sorted_unique) - 1):
                T1 = ttm_sorted_unique[i]
                T2 = ttm_sorted_unique[i+1]

                theta_T1 = self.get_ssvi_params_at_T(T1, global_ssvi_calib_params)['theta_T']
                theta_T2 = self.get_ssvi_params_at_T(T2, global_ssvi_calib_params)['theta_T']

                # Check if theta_T is decreasing
                if theta_T2 < theta_T1 - 1e-6: # Allow a small tolerance for numerical stability
                    penalty += 1e6 * np.abs(theta_T2 - theta_T1) # Penalize proportional to the decrease

        # 5. Butterfly Arbitrage Condition (related to psi_T and rho_bar):
        # A simplified condition often cited is that psi_T * (1 + |rho_bar|) < 2.
        # This condition helps ensure convexity of total variance with respect to moneyness.
        for T_val in ttm_sorted_unique:
            derived_params = self.get_ssvi_params_at_T(T_val, global_ssvi_calib_params)
            psi_T = derived_params['psi_T']

            if psi_T * (1 + np.abs(rho_bar)) >= 2.0 - 1e-6: # Allow small tolerance
                penalty += 1e5 * (psi_T * (1 + np.abs(rho_bar)) - (2.0 - 1e-6))

        # Calculate SSVI implied volatilities for all market options
        ssvi_model_vols = []
        for s, T in zip(market_strikes, market_TTMs):
            model_vol = self.ssvi_implied_volatility(s, T, global_ssvi_calib_params)
            ssvi_model_vols.append(model_vol)

        ssvi_model_vols = np.array(ssvi_model_vols)

        # Filter out NaN or Inf volatilities
        valid_vols_mask = ~np.isnan(ssvi_model_vols) & ~np.isinf(ssvi_model_vols)

        market_implied_vols_filtered = market_implied_vols[valid_vols_mask]
        ssvi_model_vols_filtered = ssvi_model_vols[valid_vols_mask]

        if not np.any(valid_vols_mask) or len(market_implied_vols_filtered) == 0:
            return np.inf # Return max penalty if no valid volatilities for comparison

        # Calculate the Sum of Squared Errors (SSE)
        errors = (ssvi_model_vols_filtered - market_implied_vols_filtered)**2
        sse = np.sum(errors)

        total_objective = sse + penalty

        # Print calibration progress
        if not np.isinf(total_objective):
            # Print only a few parameters for brevity in progress updates
            p = global_ssvi_calib_params
            print(f"  Calib. Iteration: Params=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{p[3]:.4f},{p[4]:.4f},{p[5]:.4f}], SSE={sse:.4f}, Penalty={penalty:.4f}, Total={total_objective:.4f}")
        else:
            print(f"  Calib. Iteration: Params={global_ssvi_calib_params}, SSE=INF (due to invalid vols/high penalty)")

        return total_objective

    def calibrate_ssvi_model(self, num_generations: int = 400) -> Optional[dict[str, Union[list[float], float]]]:
        """
        Calibrates the global SSVI parameters to the entire set of fetched market
        implied volatilities across all maturities. This is a single, global optimization.

        Args:
            num_generations (int, optional): The maximum number of generations for Differential Evolution.
                                             Defaults to 200.

        Returns:
            dict: A dictionary containing the calibrated global SSVI parameters and the final SSE.
                  Returns None if calibration fails.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available for calibration. Please fetch data first.")
            return None

        print("\nCalculating market implied volatilities for all fetched options (for global calibration target)...")
        # Ensure that market_option_data has 'implied_vol' calculated for all options
        self.market_option_data['implied_vol'] = self.market_option_data.apply(
            lambda row: self.implied_volatility(row['market_price'], row['strike'], row['T'], row['option_type']), axis=1
        )

        # Filter out options for which IV could not be calculated (NaNs)
        calib_data = self.market_option_data.dropna(subset=['implied_vol']).copy()

        if calib_data.empty:
            print("No valid implied volatilities could be calculated from market data for global SSVI calibration.")
            return None

        print(f"\nStarting global calibration for {self.ticker_symbol} with {len(calib_data)} valid options across {len(self.fetched_expirations)} maturities (SSVI Model)...")
        print(f"Starting model calibration (SSVI Surface) using Differential Evolution...")

        market_strikes = calib_data['strike'].values
        market_TTMs = calib_data['T'].values
        market_implied_vols = calib_data['implied_vol'].values

        # --- Initial Guess for Global SSVI Parameters ---
        # [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power]
        # These are heuristic starting points for the optimizer.
        initial_alpha0 = np.mean(market_implied_vols)**2 * np.mean(market_TTMs) * 0.5 # Avg ATM variance
        initial_alpha1 = 0.0 # Linear term for theta_T
        initial_alpha2 = 0.0 # Quadratic term for theta_T
        initial_rho_bar = -0.5 # Common for equity smirk
        initial_eta = 0.1 # Initial guess for curvature scale
        initial_gamma_power = 0.0 # Initial guess for power law exponent (often near 0 or 0.5)

        initial_guess_ssvi = [initial_alpha0, initial_alpha1, initial_alpha2,
                              initial_rho_bar, initial_eta, initial_gamma_power]

        # --- Bounds for Global SSVI Parameters ---
        # These bounds reflect commonly accepted ranges and arbitrage-free considerations.
        # [alpha0, alpha1, alpha2, rho_bar, eta, gamma_power]
        ssvi_param_bounds = [
            (1e-6, 2.0),    # alpha0 (base ATM total variance) - must be positive
            (-1.0, 1.0),    # alpha1 (linear T-dependency of ATM total variance)
            (-1.0, 1.0),    # alpha2 (quadratic T-dependency of ATM total variance)
            (-0.99, 0.99),  # rho_bar (global correlation) - strictly between -1 and 1
            (1e-6, 2.0),    # eta (global curvature scale) - must be positive
            (-1.0, 1.0)     # gamma_power (exponent for psi_T power law)
        ]

        print(f"  Global Calibration bounds: {ssvi_param_bounds}")
        print(f"  Initial Guess: {initial_guess_ssvi}")

        try:
            # Run Differential Evolution for global optimization
            result = sco.differential_evolution(
                self._objective_function_ssvi,
                bounds=ssvi_param_bounds,
                args=(market_strikes, market_TTMs, market_implied_vols), # All market data for global fit
                strategy='best1bin',
                maxiter=num_generations,
                popsize=60,
                tol=0.001,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=True,
                workers=-1,
                polish=True
            )

            # Store the calibrated global parameters if optimization was successful
            if result.x is not None and result.success:
                self.global_ssvi_params = result.x.tolist()
                print("\nSSVI Global Calibration successful!")
            else:
                print(f"\nSSVI Global Calibration terminated (failed or interrupted): {result.message if result.message else 'No specific message'}. Final SSE: {result.fun:.4f}")

            calibrated_params_dict = {
                'global_ssvi_params': self.global_ssvi_params,
                'final_sse': result.fun if hasattr(result, 'fun') else float('inf')
            }
            print("Parameters after global SSVI calibration attempt:")
            for k, v in calibrated_params_dict.items():
                print(f"  {k}: {v}")

            return calibrated_params_dict

        except Exception as e:
            print(f"\nAn error occurred during global SSVI calibration: {e}")
            return None

    # --- Local Volatility Model Specific Methods ---
    def fit_polynomial_to_implied_vol_smile(self, T_val: float, degree: int = 3):
        """
        Fits a polynomial to the implied volatility smile for a given maturity.
        This provides a smoothed representation of the implied volatility for that slice.


        Args:
            T_val (float): The time to maturity for which to fit the polynomial.
            degree (int, optional): The degree of the polynomial to fit. Defaults to 3.
                                    Higher degrees can fit more complex shapes but risk overfitting.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print(f"No market data to fit polynomial for T={T_val}.")
            return

        # Filter market data for the specific maturity and ensure implied volatilities are available
        maturity_data = self.market_option_data[self.market_option_data['T'] == T_val].dropna(subset=['implied_vol']).copy()

        if maturity_data.empty:
            print(f"No valid implied volatilities for T={T_val} to fit polynomial.")
            self.poly_coeffs_implied_vol[T_val] = np.array([np.nan] * (degree + 1))
            return

        # Use log-moneyness (k) for polynomial fitting, as smiles are often
        # more regular when plotted against log-moneyness.
        # k = ln(K/S0)
        log_moneyness_values = np.log(maturity_data['strike'].values / self.S0)
        implied_vols = maturity_data['implied_vol'].values

        # Remove any NaN or Inf from the data before fitting
        valid_indices = ~np.isnan(log_moneyness_values) & ~np.isinf(log_moneyness_values) & \
                        ~np.isnan(implied_vols) & ~np.isinf(implied_vols)

        log_moneyness_values = log_moneyness_values[valid_indices]
        implied_vols = implied_vols[valid_indices]

        if len(log_moneyness_values) <= degree:
            print(f"Not enough data points ({len(log_moneyness_values)}) for T={T_val} to fit polynomial of degree {degree}. Skipping.")
            self.poly_coeffs_implied_vol[T_val] = np.array([np.nan] * (degree + 1))
            return

        try:
            # Perform polynomial regression
            coeffs = np.polyfit(log_moneyness_values, implied_vols, degree)
            self.poly_coeffs_implied_vol[T_val] = coeffs
            print(f"  Fitted polynomial for T={T_val:.4f} (degree {degree}): {coeffs}")
        except Exception as e:
            print(f"  Error fitting polynomial for T={T_val}: {e}")
            self.poly_coeffs_implied_vol[T_val] = np.array([np.nan] * (degree + 1))


    def get_implied_vol_from_poly(self, K: float, T: float) -> float:
        """
        Retrieves the implied volatility from the fitted polynomial for a given strike K and time T.
        If no polynomial is fitted for T, returns NaN.

        Args:
            K (float): Strike price.
            T (float): Time to maturity.

        Returns:
            float: Implied volatility from the polynomial fit, or NaN.
        """
        coeffs = self.poly_coeffs_implied_vol.get(T)
        if coeffs is None or np.any(np.isnan(coeffs)):
            return np.nan

        log_moneyness = np.log(np.maximum(K / self.S0, 1e-10))

        # Evaluate the polynomial
        implied_vol = np.polyval(coeffs, log_moneyness)

        # Ensure volatility is positive and within reasonable bounds
        return np.clip(implied_vol, 1e-8, 5.0)


    def derive_local_volatility_surface(self, num_strike_points: int = 50, num_time_points: int = 50):
        """
        Derives the local volatility surface using Dupire's formula.
        This requires pre-fitting polynomials to the implied volatility smiles for each maturity.
        The local volatility is calculated on a grid of (K, T) points.

        Dupire's Formula for Local Volatility:
        sigma_local(K, T)^2 = ( sigma_imp^2 + 2T sigma_imp (d_sigma_imp/dT) + 2(r-q)K sigma_imp (d_sigma_imp/dK) ) /
                              ( (1 + K/sigma_imp * d_sigma_imp/dK)^2 + K^2 T/(2*sigma_imp) * ( d2_sigma_imp/dK2 - (d_sigma_imp/dK)^2 / sigma_imp ) )
        This formula relies on derivatives of implied volatility with respect to strike (K) and time (T).
        These derivatives are approximated using finite differences on the interpolated implied volatility surface.
        """
        if not self.poly_coeffs_implied_vol:
            print("No polynomial fits available for implied volatility. Please run `fit_polynomial_to_implied_vol_smile` first.")
            return

        print("\nDeriving Local Volatility Surface using Dupire's formula...")

        # Create a grid of strikes and maturities for local volatility
        unique_TTMs = sorted(list(self.poly_coeffs_implied_vol.keys()))
        if len(unique_TTMs) < 2:
            print("Not enough maturities to derive a robust local volatility surface. Need at least two to interpolate time derivatives.")
            return

        # Define a range for strikes and maturities for the surface
        min_strike = self.S0 * 0.7
        max_strike = self.S0 * 1.3
        strikes = np.linspace(min_strike, max_strike, num_strike_points)

        min_T = np.min(unique_TTMs)
        max_T = np.max(unique_TTMs)
        times = np.linspace(min_T, max_T, num_time_points)

        self.local_vol_surface = {} # Reset the surface

        # Helper function to get implied volatility from the polynomial interpolation
        # for any K, T (including interpolated T)
        def get_interp_implied_vol(K_val: float, T_val: float) -> float:
            # Clamp T_val to the range of unique_TTMs to avoid extrapolation issues
            clamped_T_val = np.clip(T_val, np.min(unique_TTMs), np.max(unique_TTMs))

            # Find the two closest maturities for interpolation
            if clamped_T_val in self.poly_coeffs_implied_vol:
                return self.get_implied_vol_from_poly(K_val, clamped_T_val)

            lower_T_idx = np.searchsorted(unique_TTMs, clamped_T_val) - 1
            upper_T_idx = np.searchsorted(unique_TTMs, clamped_T_val)

            # Adjust indices for boundary conditions
            if lower_T_idx < 0: lower_T_idx = 0
            if upper_T_idx >= len(unique_TTMs): upper_T_idx = len(unique_TTMs) - 1

            T_lower = unique_TTMs[lower_T_idx]
            T_upper = unique_TTMs[upper_T_idx]

            if T_lower == T_upper: # If T_val is exactly one of the known maturities
                return self.get_implied_vol_from_poly(K_val, T_lower)

            # Linear interpolation of implied volatility between maturities
            iv_lower = self.get_implied_vol_from_poly(K_val, T_lower)
            iv_upper = self.get_implied_vol_from_poly(K_val, T_upper)

            if np.isnan(iv_lower) or np.isnan(iv_upper):
                return np.nan

            interp_iv = iv_lower + (iv_upper - iv_lower) * (clamped_T_val - T_lower) / (T_upper - T_lower)
            return interp_iv

        # Small shock size for finite difference derivatives
        dK_shock = 1e-3 * self.S0
        dT_shock = 1e-3 * np.mean(times) if np.mean(times) > 1e-10 else 1e-3 # Ensure dT_shock is not zero

        for T_i in times:
            self.local_vol_surface[T_i] = {}
            for K_j in strikes:
                # Initialize derivatives to NaN to prevent NameError if conditions are not met
                ds_imp_dK = np.nan
                d2s_imp_dK2 = np.nan
                ds_imp_dT = np.nan

                # Re-evaluate sigma_imp_K for consistency and to ensure it's clamped
                sigma_imp_K = get_interp_implied_vol(K_j, T_i)

                # --- Calculate derivatives with respect to K (strike) ---
                K_plus_dk = np.maximum(K_j + dK_shock, 1e-10) # Ensure positive
                K_minus_dk = np.maximum(K_j - dK_shock, 1e-10) # Ensure positive

                sigma_imp_K_plus_dk = get_interp_implied_vol(K_plus_dk, T_i)
                sigma_imp_K_minus_dk = get_interp_implied_vol(K_minus_dk, T_i)

                # Check if values for K-derivatives are valid before calculation
                if not (np.isnan(sigma_imp_K) or np.isnan(sigma_imp_K_plus_dk) or np.isnan(sigma_imp_K_minus_dk) or dK_shock <= 1e-10):
                    ds_imp_dK = (sigma_imp_K_plus_dk - sigma_imp_K_minus_dk) / (2 * dK_shock)
                    d2s_imp_dK2 = (sigma_imp_K_plus_dk - 2 * sigma_imp_K + sigma_imp_K_minus_dk) / (dK_shock**2)
                else:
                    self.local_vol_surface[T_i][K_j] = np.nan
                    continue # Skip to next (K,T) if K-derivatives cannot be computed

                # --- Calculate derivatives with respect to T (time to maturity) ---
                T_plus_dT = T_i + dT_shock
                T_minus_dT = np.maximum(T_i - dT_shock, 1e-10) # Ensure positive and not exactly zero

                # Clamp T_plus_dT and T_minus_dT to the range of available maturities
                T_plus_dT_clamped = np.clip(T_plus_dT, np.min(unique_TTMs), np.max(unique_TTMs))
                T_minus_dT_clamped = np.clip(T_minus_dT, np.min(unique_TTMs), np.max(unique_TTMs))

                sigma_imp_T_plus_dT = get_interp_implied_vol(K_j, T_plus_dT_clamped)
                sigma_imp_T_minus_dT = get_interp_implied_vol(K_j, T_minus_dT_clamped)

                # Check if values for T-derivatives are valid before calculation
                if not (np.isnan(sigma_imp_T_plus_dT) or np.isnan(sigma_imp_T_minus_dT) or dT_shock <= 1e-10):
                    ds_imp_dT = (sigma_imp_T_plus_dT - sigma_imp_T_minus_dT) / (2 * dT_shock)
                else:
                    self.local_vol_surface[T_i][K_j] = np.nan
                    continue # Skip to next (K,T) if T-derivative cannot be computed


                # --- Compute local volatility using Dupire's formula ---
                if sigma_imp_K <= 1e-10: # Ensure sigma_imp_K is positive before using in formula
                    self.local_vol_surface[T_i][K_j] = np.nan
                    continue

                # Numerator terms
                numerator = sigma_imp_K**2 + 2 * T_i * sigma_imp_K * ds_imp_dT + 2 * (self.r - self.q) * K_j * ds_imp_dK

                # Denominator terms
                term1_denom = (1 + K_j / sigma_imp_K * ds_imp_dK)**2

                # Handle potential division by zero if sigma_imp_K is near zero for term2_denom
                term2_denom_factor = K_j**2 * T_i / (2 * sigma_imp_K) if sigma_imp_K > 1e-10 else 0.0
                term2_denom = term2_denom_factor * (d2s_imp_dK2 - (ds_imp_dK**2 / sigma_imp_K if sigma_imp_K > 1e-10 else 0.0))

                denominator = term1_denom + term2_denom

                # Ensure denominator is not zero or negative
                if denominator <= 1e-10:
                    self.local_vol_surface[T_i][K_j] = np.nan
                    continue

                local_vol_squared = numerator / denominator
                local_vol = np.sqrt(np.maximum(local_vol_squared, 1e-8)) # Ensure non-negative and clip

                self.local_vol_surface[T_i][K_j] = np.clip(local_vol, 1e-8, 5.0) # Clip local vol to reasonable bounds

        print("Local Volatility Surface derived.")


    def get_local_volatility(self, K: float, T: float) -> float:
        """
        Retrieves the local volatility for a given strike K and time T from the derived surface.
        Performs bilinear interpolation if the exact (K, T) point is not on the grid.

        Args:
            K (float): Strike price.
            T (float): Time to maturity.

        Returns:
            float: Local volatility, or NaN if not found or outside bounds.
        """
        if not self.local_vol_surface:
            return np.nan

        times_grid = sorted(list(self.local_vol_surface.keys()))
        if not times_grid: return np.nan

        # Check if the time grid contains entries for strikes
        if not all(self.local_vol_surface[t] for t in times_grid):
            # Fallback if some time slices are empty (e.g., due to derivation issues)
            print("Warning: Some time slices in local_vol_surface are empty. Cannot interpolate.")
            return np.nan

        strikes_grid = sorted(list(self.local_vol_surface[times_grid[0]].keys()))
        if not strikes_grid: return np.nan

        # If T or K are outside the grid, return NaN
        if T < times_grid[0] or T > times_grid[-1] or K < strikes_grid[0] or K > strikes_grid[-1]:
            return np.nan

        # Find the bounding grid points for T
        t_idx = np.searchsorted(times_grid, T)
        # Ensure t1_idx is within bounds and t2_idx is valid.
        # If T is less than the first TTM, t_idx will be 0. We'll use the first TTM.
        # If T is greater than the last TTM, t_idx will be len(times_grid). We'll use the last TTM.
        t1_idx = max(0, t_idx - 1)
        t2_idx = min(len(times_grid) - 1, t_idx)

        T1 = times_grid[t1_idx]
        T2 = times_grid[t2_idx]

        # If T is exactly on a grid point
        if T1 == T2:
            return self._interpolate_strike_slice(K, T1, strikes_grid, self.local_vol_surface)

        # Find the bounding grid points for K
        k_idx = np.searchsorted(strikes_grid, K)
        k1_idx = max(0, k_idx - 1)
        k2_idx = min(len(strikes_grid) - 1, k_idx)

        K1 = strikes_grid[k1_idx]
        K2 = strikes_grid[k2_idx]

        # If K is exactly on a grid point (for T1 and T2)
        if K1 == K2:
            lv_T1 = self.local_vol_surface[T1].get(K1)
            lv_T2 = self.local_vol_surface[T2].get(K1)
            if np.isnan(lv_T1) or np.isnan(lv_T2): return np.nan
            # Linear interpolation for T
            return lv_T1 + (lv_T2 - lv_T1) * (T - T1) / (T2 - T1)

        # Bilinear interpolation
        lv_Q11 = self.local_vol_surface[T1].get(K1)
        lv_Q12 = self.local_vol_surface[T1].get(K2)
        lv_Q21 = self.local_vol_surface[T2].get(K1)
        lv_Q22 = self.local_vol_surface[T2].get(K2)

        if np.isnan(lv_Q11) or np.isnan(lv_Q12) or np.isnan(lv_Q21) or np.isnan(lv_Q22):
            return np.nan

        # Interpolate in K at T1 and T2
        lv_at_T1 = lv_Q11 * (K2 - K) / (K2 - K1) + lv_Q12 * (K - K1) / (K2 - K1)
        lv_at_T2 = lv_Q21 * (K2 - K) / (K2 - K1) + lv_Q22 * (K - K1) / (K2 - K1)

        # Interpolate in T
        final_lv = lv_at_T1 * (T2 - T) / (T2 - T1) + lv_at_T2 * (T - T1) / (T2 - T1)

        return np.clip(final_lv, 1e-8, 5.0)

    def _interpolate_strike_slice(self, K: float, T: float, strikes_grid: list[float], surface: dict[float, dict[float, float]]) -> float:
        """Helper to interpolate along a single strike slice of a surface."""
        slice_data = surface[T]
        if K in slice_data:
            return slice_data[K]

        k_idx = np.searchsorted(strikes_grid, K)
        k1_idx = max(0, k_idx - 1)
        k2_idx = min(len(strikes_grid) - 1, k_idx)

        K1 = strikes_grid[k1_idx]
        K2 = strikes_grid[k2_idx]

        if K1 == K2: # K is exactly on a grid line
            return slice_data.get(K1, np.nan)

        lv_K1 = slice_data.get(K1)
        lv_K2 = slice_data.get(K2)

        if np.isnan(lv_K1) or np.isnan(lv_K2): return np.nan

        interp_val = lv_K1 + (lv_K2 - lv_K1) * (K - K1) / (K2 - K1)
        return interp_val

    # --- SABR Model Specific Methods (New) ---

    def _sabr_implied_volatility_hagan(self, K: float, F: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
        """
        Calculates the Black-Scholes implied volatility using Hagan's 2002 SABR approximation.


        Args:
            K (float): Strike price.
            F (float): Forward price of the underlying asset.
            T (float): Time to maturity (in years).
            alpha (float): SABR alpha parameter (ATM volatility level).
            beta (float): SABR beta parameter (elasticity of volatility).
            rho (float): SABR rho parameter (correlation between asset price and volatility).
            nu (float): SABR nu parameter (volatility of volatility).

        Returns:
            float: Black-Scholes implied volatility. Returns NaN if inputs are invalid.
        """
        # Ensure parameters are within reasonable bounds to prevent numerical issues
        alpha = np.maximum(alpha, 1e-8) # alpha must be positive
        beta = np.clip(beta, 0.0 + 1e-8, 1.0 - 1e-8) # beta in (0, 1)
        rho = np.clip(rho, -1.0 + 1e-8, 1.0 - 1e-8) # rho in (-1, 1)
        nu = np.maximum(nu, 1e-8) # nu must be positive

        if T <= 1e-10: return np.nan # Avoid division by zero for T
        if F <= 1e-10: return np.nan # Avoid division by zero for F (forward)

        # Handle special case where K is very close to F (ATM) to avoid division by zero
        if np.isclose(F, K, atol=1e-6): # Use np.isclose for floating point comparison
            # ATM approximation for SABR
            V_atm = F**(beta - 1) * alpha * (
                1 + (nu**2 / 24) * T + (alpha**2 / (24 * F**(2 - 2 * beta))) * T + (rho * nu * alpha / (4 * F**(1 - beta))) * T
            )
            return np.maximum(V_atm, 1e-8) # Ensure volatility is positive

        # General Hagan's formula for K != F
        log_FK = np.log(F / K)
        FK_beta = F * K
        FK_beta_pow = FK_beta**(1 - beta)

        # Z term
        if np.isclose(beta, 1.0): # Handle beta=1 special case for X
            X_val = log_FK
        else:
            X_val = (F**(1 - beta) - K**(1 - beta)) / (1 - beta)

        # Avoid division by zero if nu is effectively zero
        if nu == 0:
            Z = X_val / (alpha * FK_beta_pow) # Simplified Z for nu=0
        else:
            Z = (nu / alpha) * X_val / FK_beta_pow

        # Guard against log(0) for Xi, this happens if Z is too small/zero and sqrt_term_arg is near (1-rho^2)
        sqrt_term_arg = 1 - 2 * rho * Z + Z**2
        if sqrt_term_arg <= 0: return np.nan # Argument must be positive

        # Guard against division by zero in Xi calculation
        if np.isclose(Z, rho) and np.isclose(Z, 1) and np.isclose(rho, 1):
             # This is a complex singular point, return nan or a very high vol
            return np.nan

        Xi = np.log((np.sqrt(sqrt_term_arg) + Z - rho) / (1 - rho))

        # Denominator (sigma_bar_factor)
        sigma_bar_factor_arg = F**(1 - beta)
        if sigma_bar_factor_arg <= 1e-10: return np.nan # Avoid division by zero
        sigma_bar_factor = alpha / sigma_bar_factor_arg

        # Correction terms (terms_correction)
        term1 = (beta - 1)**2 / 24 * T
        term2 = rho * beta * nu / 4 * T
        term3 = (nu**2) / 24 * T

        terms_correction = 1 + (term1 + term2 + term3)

        # Final implied volatility
        if np.isclose(Xi, 0, atol=1e-8) or np.isnan(Xi): # If Xi is near zero or NaN, use L'Hopital's rule for Z/Xi
            implied_vol = sigma_bar_factor * terms_correction # simplified form for small Z
        else:
            implied_vol = sigma_bar_factor * (Z / Xi) * terms_correction

        # Ensure volatility is positive and within reasonable bounds
        return np.clip(implied_vol, 1e-8, 5.0)


    def _objective_function_sabr(self, sabr_params: list[float], strikes: np.ndarray,
                                 market_implied_vols: np.ndarray, forward_price: float, T_val: float) -> np.ndarray:
        """
        Objective function for SABR calibration (used by least_squares).
        Calculates the residuals (difference) between market implied volatilities
        and SABR model-predicted implied volatilities for a single maturity.

        Args:
            sabr_params (list): SABR parameters [alpha, beta, rho, nu].
            strikes (np.ndarray): Array of strike prices for the current maturity.
            market_implied_vols (np.ndarray): Array of market implied volatilities for the current maturity.
            forward_price (float): Forward price for the current maturity.
            T_val (float): Time to maturity.

        Returns:
            np.ndarray: Array of residuals (market_vol - model_vol).
        """
        alpha, beta, rho, nu = sabr_params

        model_vols = np.array([
            self._sabr_implied_volatility_hagan(K, forward_price, T_val, alpha, beta, rho, nu)
            for K in strikes
        ])

        # Calculate residuals, ignoring NaNs
        residuals = market_implied_vols - model_vols

        # Penalize NaNs or Inf values heavily
        residuals[np.isnan(residuals)] = 100.0 # Arbitrary large penalty
        residuals[np.isinf(residuals)] = 100.0

        # Add penalties for parameter bounds if least_squares doesn't handle them perfectly
        # (though least_squares with bounds is generally robust)
        penalty = 0.0
        if alpha <= 0 or nu <= 0: penalty += 100
        if not (-0.999 < rho < 0.999): penalty += 100
        if not (1e-8 < beta < 1.0): penalty += 100 # Beta should be in (0,1)

        return np.sqrt(residuals**2 + penalty) # Return square root of sum of squares for least_squares


    def calibrate_sabr_model(self) -> None:
        """
        Calibrates the SABR model parameters for each fetched maturity slice.
        Stores the calibrated parameters in `self.sabr_params_per_maturity`.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available for SABR calibration. Please fetch data first.")
            return

        print("\n--- Starting SABR Model Calibration (Per Maturity) ---")

        # Ensure implied volatilities are already calculated
        if 'implied_vol' not in self.market_option_data.columns or self.market_option_data['implied_vol'].isnull().all():
             self.market_option_data['implied_vol'] = self.market_option_data.apply(
                lambda row: self.implied_volatility(row['market_price'], row['strike'], row['T'], row['option_type']), axis=1
            )
             self.market_option_data.dropna(subset=['implied_vol'], inplace=True)
             if self.market_option_data.empty:
                 print("No valid implied volatilities found for SABR calibration after initial calculation.")
                 return


        unique_TTMs = sorted(self.market_option_data['T'].unique())
        if not unique_TTMs:
            print("No unique maturities found for SABR calibration.")
            return

        self.sabr_params_per_maturity = {} # Reset SABR parameters

        for T_val in unique_TTMs:
            maturity_data = self.market_option_data[self.market_option_data['T'] == T_val].dropna(subset=['implied_vol']).copy()

            if maturity_data.empty or len(maturity_data) < 4: # Need at least 4 points to fit 4 parameters
                print(f"  Skipping SABR calibration for T={T_val:.4f}: Not enough valid data points ({len(maturity_data)}).")
                self.sabr_params_per_maturity[T_val] = [np.nan] * 4 # Store NaNs if not calibrated
                continue

            strikes = maturity_data['strike'].values
            market_vols = maturity_data['implied_vol'].values

            # Calculate forward price for the current maturity
            forward_price = self.S0 * np.exp((self.r - self.q) * T_val)

            # --- Initial Guesses for SABR parameters [alpha, beta, rho, nu] ---
            # Try to infer reasonable initial guesses from market data
            # Alpha: Approximately ATM implied volatility
            alpha_init = 0.2
            alpha_init = np.clip(alpha_init, 0.05, 0.5) # Clamp to reasonable vol range

            # Beta: Often fixed at 0.5 for equities, but can be calibrated (0, 1)
            beta_init = 0.5

            # Rho: Often negative for equity (smirk), between -1 and 1
            rho_init = -0.5

            # Nu: Volatility of volatility, positive. Often between 0 and 2.
            nu_init = 0.5

            initial_guess = [alpha_init, beta_init, rho_init, nu_init]

            # --- Bounds for SABR parameters ---
            # [alpha, beta, rho, nu]
            bounds = [
                (1e-8, 2.0),   # alpha (must be positive, up to 200%)
                (0.01, 0.99),  # beta (strictly between 0 and 1)
                (-0.99, 0.99), # rho (strictly between -1 and 1)
                (1e-8, 2.0)    # nu (must be positive, up to 200% vol-of-vol)
            ]
            bounds_min = [1e-8, 0.01, -0.99, 1e-8]
            bounds_max = [2.0, 0.99, 0.99, 2.0]
            initial_guess = np.clip(initial_guess, bounds_min, bounds_max)

            try:
                # Use least_squares for robust non-linear optimization with bounds
                # Pass extra arguments (strikes, market_vols, forward_price, T_val)
                result = sco.least_squares(
                    self._objective_function_sabr,
                    initial_guess,
                    bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                    args=(strikes, market_vols, forward_price, T_val),
                    method='trf', # Trust-Region-Reflective algorithm for bounded problems
                    xtol=1e-8, # Tolerance for parameter changes
                    ftol=1e-8, # Tolerance for function value changes
                    max_nfev=5000 # Max function evaluations
                )

                if result.success:
                    calibrated_sabr_params = result.x.tolist()
                    self.sabr_params_per_maturity[T_val] = calibrated_sabr_params
                    print(f"  Calibrated SABR for T={T_val:.4f}: alpha={calibrated_sabr_params[0]:.4f}, beta={calibrated_sabr_params[1]:.4f}, rho={calibrated_sabr_params[2]:.4f}, nu={calibrated_sabr_params[3]:.4f}")
                else:
                    print(f"  SABR calibration failed for T={T_val:.4f}: {result.message}. Storing NaNs.")
                    self.sabr_params_per_maturity[T_val] = [np.nan] * 4 # Store NaNs if calibration fails
            except Exception as e:
                print(f"  Error during SABR calibration for T={T_val:.4f}: {e}. Storing NaNs.")
                self.sabr_params_per_maturity[T_val] = [np.nan] * 4 # Store NaNs if an error occurs

        print("SABR calibration completed for all maturities.")


    # --- Common Calibration and Greeks Orchestration ---
    def calibrate_all_models(self, run_ssvi: bool = True, run_local_vol: bool = True,
                             run_sabr: bool = True, num_generations: int = 200, poly_degree: int = 3):
        """
        Orchestrates the calibration of all implemented volatility surface models.

        Args:
            run_ssvi (bool, optional): Whether to run SSVI global calibration. Defaults to True.
            run_local_vol (bool, optional): Whether to run Local Volatility surface derivation. Defaults to True.
            run_sabr (bool, optional): Whether to run SABR per-maturity calibration. Defaults to True.
            num_generations (int, optional): Generations for SSVI Differential Evolution. Defaults to 200.
            poly_degree (int, optional): Degree of polynomial for Local Volatility smile fitting. Defaults to 3.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available for calibration. Please fetch data first.")
            return

        print("\n--- Starting Calibration of All Volatility Models ---")

        # First, ensure implied volatilities are calculated for all market options
        # This is a prerequisite for all model calibrations.
        self.market_option_data['implied_vol'] = self.market_option_data.apply(
            lambda row: self.implied_volatility(row['market_price'], row['strike'], row['T'], row['option_type']), axis=1
        )
        # Drop options where IV couldn't be computed
        self.market_option_data.dropna(subset=['implied_vol'], inplace=True)
        if self.market_option_data.empty:
            print("No valid implied volatilities found after filtering. Cannot proceed with calibration.")
            return

        if run_ssvi:
            self.calibrate_ssvi_model(num_generations=num_generations)
            if self.global_ssvi_params is None:
                print("\nWARNING: SSVI global calibration did not yield valid parameters.")

        if run_local_vol:
            print("\n--- Starting Local Volatility Surface Derivation ---")
            unique_TTMs = sorted(self.market_option_data['T'].unique())
            if len(unique_TTMs) < 2:
                print("Not enough maturities for meaningful local volatility derivation. Need at least two.")
            else:
                for T_val in unique_TTMs:
                    print(f"  Fitting polynomial for implied vol smile at T={T_val:.4f}...")
                    self.fit_polynomial_to_implied_vol_smile(T_val, degree=poly_degree)

                # Derive the local volatility surface once all polynomials are fitted
                self.derive_local_volatility_surface()
                if not self.local_vol_surface:
                    print("\nWARNING: Local Volatility surface could not be derived.")

        if run_sabr:
            self.calibrate_sabr_model()
            if not self.sabr_params_per_maturity:
                print("\nWARNING: SABR calibration did not yield valid parameters for any maturity.")


    def get_model_implied_volatility(self, K: float, T: float, model_type: str = 'SSVI', S0_override: Optional[float] = None, r_override: Optional[float] = None, q_override: Optional[float] = None) -> float:
        """
        Retrieves implied volatility from either the SSVI, Local Volatility, or SABR model.
        Includes overrides for S0, r, q for Monte Carlo/Greek calculations for consistency.

        Args:
            K (float): Strike price.
            T (float): Time to maturity.
            model_type (str, optional): 'SSVI', 'LocalVol', or 'SABR'. Defaults to 'SSVI'.
            S0_override (float, optional): Override for the underlying asset price.
            r_override (float, optional): Optional override for the risk-free rate.
            q_override (float, optional): Optional override for the dividend yield.

        Returns:
            float: Implied volatility from the specified model, or NaN if model not calibrated/data unavailable.
        """
        current_S = S0_override if S0_override is not None else self.S0
        current_r = r_override if r_override is not None else self.r
        current_q = q_override if q_override is not None else self.q

        if model_type == 'SSVI':
            if self.global_ssvi_params is None:
                return np.nan
            return self.ssvi_implied_volatility(K, T, self.global_ssvi_params, S0_override=current_S)
        elif model_type == 'LocalVol':
            return self.get_implied_vol_from_poly(K, T) # LocalVol direct IV from poly fit
        elif model_type == 'SABR':
            sabr_params = self.sabr_params_per_maturity.get(T)
            if sabr_params is None or np.any(np.isnan(sabr_params)):
                # If exact T not found, try to interpolate SABR parameters for nearby T
                unique_TTMs = sorted(list(self.sabr_params_per_maturity.keys()))
                if len(unique_TTMs) < 2: return np.nan

                # Find bounding maturities for interpolation
                lower_T_idx = np.searchsorted(unique_TTMs, T) - 1
                upper_T_idx = np.searchsorted(unique_TTMs, T)

                T_lower = unique_TTMs[max(0, lower_T_idx)]
                T_upper = unique_TTMs[min(len(unique_TTMs) - 1, upper_T_idx)]

                if T_lower == T_upper: # If T is exactly on a calibrated maturity or outside range
                    sabr_params = self.sabr_params_per_maturity.get(T_lower) # Use the closest one
                    if sabr_params is None or np.any(np.isnan(sabr_params)): return np.nan
                else: # Interpolate parameters if T is between calibrated maturities
                    sabr_params_lower = self.sabr_params_per_maturity.get(T_lower)
                    sabr_params_upper = self.sabr_params_per_maturity.get(T_upper)

                    if sabr_params_lower is None or sabr_params_upper is None or np.any(np.isnan(sabr_params_lower)) or np.any(np.isnan(sabr_params_upper)):
                        return np.nan

                    # Linear interpolation of SABR parameters
                    interp_factor = (T - T_lower) / (T_upper - T_lower)
                    sabr_params = [
                        p_low + (p_high - p_low) * interp_factor
                        for p_low, p_high in zip(sabr_params_lower, sabr_params_upper)
                    ]

            # Calculate forward price for SABR IV using potentially overridden S0, r, q
            forward_price = current_S * np.exp((current_r - current_q) * T)
            return self._sabr_implied_volatility_hagan(K, forward_price, T, *sabr_params)
        else:
            raise ValueError("model_type must be 'SSVI', 'LocalVol', or 'SABR'")


    def calculate_greeks(self, S_val: float, K_val: float, T_val: float, r_val: float, q_val: float,
                         option_type: str = 'call', model_type: str = 'SSVI', shock_size: float = 1e-4) -> dict[str, float]:
        """
        Calculates option Greeks (Delta, Gamma, Vega, Theta, Rho) for a given option
        using finite difference approximations, leveraging either the SSVI, Local Volatility, or SABR model.

        Args:
            S_val (float): Underlying asset price at which to calculate Greeks.
            K_val (float): Strike price of the option.
            T_val (float): Time to maturity of the option (in years).
            r_val (float): Risk-free rate.
            q_val (float): Dividend yield.
            option_type (str, optional): 'call' or 'put'. Defaults to 'call'.
            model_type (str, optional): 'SSVI', 'LocalVol', or 'SABR'. Specifies which model's volatility to use.
                                        Defaults to 'SSVI'.
            shock_size (float, optional): The relative perturbation size for finite differences (e.g., 1e-4 for 0.01%).
                                          Defaults to 1e-4.

        Returns:
            dict: A dictionary containing the calculated Greeks (Delta, Gamma, Vega, Theta, Rho).
                  Returns NaN for any Greek that cannot be calculated.
        """
        greeks = {greek: np.nan for greek in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']}

        # Define a helper pricing function that uses the specified model's implied volatility.
        # This function will pass the perturbed S, r, q values to `get_model_implied_volatility`.
        def price_func_for_greeks(S_arg: float, K_arg: float, T_arg: float, r_arg: float, q_arg: float, type_arg: str, current_model_type: str) -> float:
            sigma_to_use = self.get_model_implied_volatility(
                K=K_arg,
                T=T_arg,
                model_type=current_model_type,
                S0_override=S_arg, # Pass perturbed S for moneyness in SSVI/SABR forward
                r_override=r_arg,  # Pass perturbed r for SABR forward
                q_override=q_arg   # Pass perturbed q for SABR forward
            )

            if np.isnan(sigma_to_use) or sigma_to_use <= 1e-8:
                return np.nan

            return self.black_scholes_price(S_arg, K_arg, T_arg, r_arg, q_arg, sigma_to_use, type_arg)

        # --- Delta (Sensitivity to underlying asset price S) ---
        dS = S_val * shock_size
        S_plus = np.maximum(S_val + dS, 1e-6)
        S_minus = np.maximum(S_val - dS, 1e-6)

        try:
            price_plus_S = price_func_for_greeks(S_plus, K_val, T_val, r_val, q_val, option_type, model_type)
            price_minus_S = price_func_for_greeks(S_minus, K_val, T_val, r_val, q_val, option_type, model_type)

            greeks['Delta'] = (price_plus_S - price_minus_S) / (2 * dS)
        except Exception as e:
            # print(f"  Error calculating Delta for {model_type}: {e}")
            pass

        # --- Gamma (Second sensitivity to underlying asset price S) ---
        try:
            price_S_current = price_func_for_greeks(S_val, K_val, T_val, r_val, q_val, option_type, model_type)
            gamma = (price_plus_S - 2 * price_S_current + price_minus_S) / (dS**2)
            greeks['Gamma'] = gamma
        except Exception as e:
            # print(f"  Error calculating Gamma for {model_type}: {e}")
            pass

        # --- Vega (Sensitivity to volatility level) ---
        # For simplicity and general applicability, Vega is defined as the sensitivity to a 1% parallel shift in the implied volatility curve at this strike and maturity.
        try:
            base_iv = self.get_model_implied_volatility(K_val, T_val, model_type=model_type, S0_override=S_val, r_override=r_val, q_override=q_val)
            if np.isnan(base_iv) or base_iv <= 1e-8:
                greeks['Vega'] = np.nan
            else:
                d_iv_shock = base_iv * shock_size # Relative shock to current IV
                iv_plus = np.maximum(base_iv + d_iv_shock, 1e-8)
                iv_minus = np.maximum(base_iv - d_iv_shock, 1e-8)

                price_plus_iv = self.black_scholes_price(S_val, K_val, T_val, r_val, q_val, iv_plus, option_type)
                price_minus_iv = self.black_scholes_price(S_val, K_val, T_val, r_val, q_val, iv_minus, option_type)

                vega = (price_plus_iv - price_minus_iv) / (2 * d_iv_shock) * 0.01 # Per 1% vol change
                greeks['Vega'] = vega

        except Exception as e:
            # print(f"  Error calculating Vega for {model_type}: {e}")
            pass

        # --- Theta (Sensitivity to time T) ---
        dT = 1/365.25
        T_plus = T_val + dT
        T_minus = np.maximum(T_val - dT, 1e-10)

        try:
            price_plus_T = price_func_for_greeks(S_val, K_val, T_plus, r_val, q_val, option_type, model_type)
            price_minus_T = price_func_for_greeks(S_val, K_val, T_minus, r_val, q_val, option_type, model_type)

            theta = (price_plus_T - price_minus_T) / (2 * dT) * (-1/365.25)
            greeks['Theta'] = theta
        except Exception as e:
            # print(f"  Error calculating Theta for {model_type}: {e}")
            pass

        # --- Rho (Sensitivity to risk-free rate r) ---
        dr = r_val * shock_size
        r_plus = np.maximum(r_val + dr, 1e-8)
        r_minus = np.maximum(r_val - dr, 1e-8)

        try:
            price_plus_r = price_func_for_greeks(S_val, K_val, T_val, r_plus, q_val, option_type, model_type)
            price_minus_r = price_func_for_greeks(S_val, K_val, T_val, r_minus, q_val, option_type, model_type)

            rho = (price_plus_r - price_minus_r) / (2 * dr) * 0.01
            greeks['Rho'] = rho
        except Exception as e:
            # print(f"  Error calculating Rho for {model_type}: {e}")
            pass


        # For puts, convert call Greeks using Put-Call Parity relations
        if option_type == 'put':
            # Apply parity adjustments if base price is valid
            base_price = price_func_for_greeks(S_val, K_val, T_val, r_val, q_val, option_type, model_type)
            if not np.isnan(base_price):
                 # Only apply if Delta is not NaN
                if not np.isnan(greeks['Delta']):
                    greeks['Delta'] = greeks['Delta'] - np.exp(-q_val * T_val)
                # Theta and Rho parity adjustments
                greeks['Theta'] = greeks['Theta'] + q_val * S_val * np.exp(-q_val * T_val) - r_val * K_val * np.exp(-r_val * T_val)
                greeks['Rho'] = greeks['Rho'] - K_val * T_val * np.exp(-r_val * T_val)

        return greeks

    # --- Monte Carlo Simulation for Exotic Options (New) ---
    def monte_carlo_asian_option_price(self, K: float, T: float, option_type: str = 'call',
                                       model_type: str = 'LocalVol', num_paths: int = 10000, num_steps: int = 100) -> float:
        """
        Prices an Arithmetic Mean Asian Option (fixed strike) using Monte Carlo simulation.
        The instantaneous volatility for path generation is derived from the chosen volatility surface model.

        Args:
            K (float): Strike price of the Asian option.
            T (float): Time to maturity (in years).
            option_type (str, optional): 'call' or 'put'. Defaults to 'call'.
            model_type (str, optional): The volatility model to use for path generation:
                                        'LocalVol', 'SSVI', or 'SABR'. Defaults to 'LocalVol'.
                                        'LocalVol' is generally preferred for MC as it provides instantaneous volatility.
            num_paths (int, optional): Number of Monte Carlo simulation paths. Defaults to 10000.
            num_steps (int, optional): Number of time steps in each path. Defaults to 100.

        Returns:
            float: The Monte Carlo estimated price of the Asian option.
        """
        print(f"\nStarting Monte Carlo simulation for Asian Option (K={K:.2f}, T={T:.4f}, Type={option_type}, Model={model_type})...")
        print(f"  Simulating {num_paths} paths with {num_steps} steps.")

        dt = T / num_steps

        # Array to store payoffs from each path
        payoffs = np.zeros(num_paths)

        # Loop through each simulation path
        for i in range(num_paths):
            current_S = self.S0
            sum_S_for_average = self.S0 # Include initial price in average

            # Simulate each step along the path
            for j in range(1, num_steps + 1): # Start from 1 to include all steps
                t_elapsed = j * dt
                t_remaining = T - t_elapsed
                t_remaining = np.maximum(t_remaining, 1e-10) # Ensure T is positive for vol lookup

                # Determine instantaneous volatility from the chosen surface model
                sigma_inst = np.nan

                if model_type == 'LocalVol':
                    if self.local_vol_surface:
                        # For Local Vol, the instantaneous volatility is directly available.
                        # We use the current stock price (S) and remaining time (T) as K and T for lookup.
                        sigma_inst = self.get_local_volatility(current_S, t_remaining)
                elif model_type == 'SSVI':
                    if self.global_ssvi_params is not None:
                        # For SSVI, get implied vol at current S (as strike) and remaining T.
                        # This serves as a proxy for instantaneous volatility.
                        sigma_inst = self.ssvi_implied_volatility(current_S, t_remaining, self.global_ssvi_params, S0_override=current_S)
                elif model_type == 'SABR':
                    if self.sabr_params_per_maturity:
                        # For SABR, get implied vol at current S (as strike) and remaining T.
                        # This also serves as a proxy for instantaneous volatility.
                        # Need to provide current S as S0_override for forward calculation within get_model_implied_volatility
                        sigma_inst = self.get_model_implied_volatility(current_S, t_remaining, model_type='SABR', S0_override=current_S, r_override=self.r, q_override=self.q)
                else:
                    print("  Invalid model_type specified for Monte Carlo. Using flat 20% vol.")
                    sigma_inst = 0.2

                # Fallback if specific model's volatility is NaN or invalid
                if np.isnan(sigma_inst) or sigma_inst <= 1e-8:
                    sigma_inst = 0.2 # Fallback to a reasonable default volatility if lookup fails
                    # print(f"    Warning: Volatility lookup failed at step {j} (T_rem={t_remaining:.4f}, S={current_S:.2f}). Using fallback vol: {sigma_inst:.2f}")

                # Generate random number from standard normal distribution
                z = np.random.normal(0, 1)

                # Update stock price using Geometric Brownian Motion
                # The volatility `sigma_inst` is now dependent on (current_S, t_remaining)
                current_S *= np.exp((self.r - self.q - 0.5 * sigma_inst**2) * dt + sigma_inst * np.sqrt(dt) * z)
                current_S = np.maximum(current_S, 1e-6) # Ensure price is always positive

                sum_S_for_average += current_S

            # Calculate average price over the path
            average_S_path = sum_S_for_average / (num_steps + 1) # Include initial price and num_steps prices

            # Calculate payoff for the Asian option
            if option_type == 'call':
                payoff = np.maximum(average_S_path - K, 0)
            elif option_type == 'put':
                payoff = np.maximum(K - average_S_path, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            payoffs[i] = payoff

        # Calculate the estimated option price by averaging payoffs and discounting
        estimated_price = np.mean(payoffs) * np.exp(-self.r * T)

        print(f"  Monte Carlo Asian Option Price estimated: {estimated_price:.4f}")
        return estimated_price

    def monte_carlo_barrier_option_price(self, K: float, T: float, barrier: float, barrier_type: str,
                                         option_type: str = 'call', model_type: str = 'LocalVol',
                                         num_paths: int = 10000, num_steps: int = 252) -> float:
        """
        Prices a Barrier Option using Monte Carlo simulation.
        Supports common barrier types: 'up_and_out_call', 'up_and_in_call',
        'down_and_out_call', 'down_and_in_call', and their put equivalents.

        Args:
            K (float): Strike price.
            T (float): Time to maturity (in years).
            barrier (float): Barrier level.
            barrier_type (str): Type of barrier option (e.g., 'up_and_out_call', 'down_and_in_put').
            option_type (str, optional): 'call' or 'put'. This should align with barrier_type. Defaults to 'call'.
            model_type (str, optional): Volatility model for path generation. Defaults to 'LocalVol'.
            num_paths (int, optional): Number of simulation paths. Defaults to 10000.
            num_steps (int, optional): Number of time steps. Defaults to 252 (approx. daily for 1 year).

        Returns:
            float: Monte Carlo estimated price of the barrier option.
        """
        print(f"\nStarting Monte Carlo simulation for Barrier Option (K={K:.2f}, T={T:.4f}, Barrier={barrier:.2f}, Type={barrier_type}, Model={model_type})...")
        print(f"  Simulating {num_paths} paths with {num_steps} steps.")

        dt = T / num_steps
        payoffs = np.zeros(num_paths)

        # Validate barrier type
        valid_barrier_types = ['up_and_out_call', 'up_and_in_call',
                               'down_and_out_call', 'down_and_in_call',
                               'up_and_out_put', 'up_and_in_put',
                               'down_and_out_put', 'down_and_in_put']
        if barrier_type not in valid_barrier_types:
            raise ValueError(f"Invalid barrier_type: {barrier_type}. Must be one of {valid_barrier_types}")

        for i in range(num_paths):
            current_S = self.S0
            path_active = True # Flag to track if the option is still active (not knocked out)
            path_hit_barrier = False # Flag to track if the barrier was hit

            # Simulate each step along the path
            for j in range(num_steps):
                t_remaining = T - (j * dt)
                t_remaining = np.maximum(t_remaining, 1e-10)

                sigma_inst = np.nan
                # Fetch instantaneous volatility from the chosen surface
                if model_type == 'LocalVol':
                    if self.local_vol_surface:
                        sigma_inst = self.get_local_volatility(current_S, t_remaining)
                elif model_type == 'SSVI':
                    if self.global_ssvi_params is not None:
                        sigma_inst = self.ssvi_implied_volatility(current_S, t_remaining, self.global_ssvi_params, S0_override=current_S)
                elif model_type == 'SABR':
                    if self.sabr_params_per_maturity:
                        sigma_inst = self.get_model_implied_volatility(current_S, t_remaining, model_type='SABR', S0_override=current_S, r_override=self.r, q_override=self.q)

                if np.isnan(sigma_inst) or sigma_inst <= 1e-8:
                    sigma_inst = 0.2 # Fallback

                z = np.random.normal(0, 1)

                # Step the price
                next_S = current_S * np.exp((self.r - self.q - 0.5 * sigma_inst**2) * dt + sigma_inst * np.sqrt(dt) * z)
                next_S = np.maximum(next_S, 1e-6) # Ensure price is positive

                # --- Barrier Monitoring (Continuous Approximation) ---
                # Check if barrier was crossed during this step.
                # Use max/min of current_S and next_S to approximate continuous monitoring.
                max_price_in_step = np.maximum(current_S, next_S)
                min_price_in_step = np.minimum(current_S, next_S)

                if 'up_and_out' in barrier_type:
                    if max_price_in_step >= barrier:
                        path_hit_barrier = True
                        path_active = False # Knocked out
                        break # Path ends here if knocked out
                elif 'down_and_out' in barrier_type:
                    if min_price_in_step <= barrier:
                        path_hit_barrier = True
                        path_active = False # Knocked out
                        break
                elif 'up_and_in' in barrier_type:
                    if max_price_in_step >= barrier:
                        path_hit_barrier = True # Knocked in
                elif 'down_and_in' in barrier_type:
                    if min_price_in_step <= barrier:
                        path_hit_barrier = True # Knocked in

                current_S = next_S

            # Calculate final payoff based on barrier type and path activity
            final_payoff = 0.0
            terminal_S = current_S # S at maturity

            if 'out' in barrier_type: # Knock-out option
                if path_active: # If not knocked out
                    if option_type == 'call':
                        final_payoff = np.maximum(terminal_S - K, 0)
                    elif option_type == 'put':
                        final_payoff = np.maximum(K - terminal_S, 0)
            elif 'in' in barrier_type: # Knock-in option
                if path_hit_barrier: # If knocked in
                    if option_type == 'call':
                        final_payoff = np.maximum(terminal_S - K, 0)
                    elif option_type == 'put':
                        final_payoff = np.maximum(K - terminal_S, 0)

            payoffs[i] = final_payoff

        estimated_price = np.mean(payoffs) * np.exp(-self.r * T)
        print(f"  Monte Carlo Barrier Option Price estimated: {estimated_price:.4f}")
        return estimated_price


    def monte_carlo_lookback_option_price(self, K: Optional[float], T: float, lookback_type: str,
                                          option_type: str = 'call', model_type: str = 'LocalVol',
                                          num_paths: int = 10000, num_steps: int = 252) -> float:
        """
        Prices a Lookback Option using Monte Carlo simulation.
        Supports fixed and floating strike lookback options.

        Args:
            K (float, optional): Strike price. Required for fixed strike, None for floating strike.
            T (float): Time to maturity (in years).
            lookback_type (str): Type of lookback option ('fixed_strike_call', 'fixed_strike_put',
                                 'floating_strike_call', 'floating_strike_put').
            option_type (str, optional): 'call' or 'put'. Should align with lookback_type. Defaults to 'call'.
            model_type (str, optional): Volatility model for path generation. Defaults to 'LocalVol'.
            num_paths (int, optional): Number of simulation paths. Defaults to 10000.
            num_steps (int, optional): Number of time steps. Defaults to 252.

        Returns:
            float: Monte Carlo estimated price of the lookback option.
        """
        print(f"\nStarting Monte Carlo simulation for Lookback Option (K={K if K is not None else 'N/A'}, T={T:.4f}, Type={lookback_type}, Model={model_type})...")
        print(f"  Simulating {num_paths} paths with {num_steps} steps.")

        dt = T / num_steps
        payoffs = np.zeros(num_paths)

        valid_lookback_types = ['fixed_strike_call', 'fixed_strike_put',
                                'floating_strike_call', 'floating_strike_put']
        if lookback_type not in valid_lookback_types:
            raise ValueError(f"Invalid lookback_type: {lookback_type}. Must be one of {valid_lookback_types}")

        if 'fixed_strike' in lookback_type and K is None:
            raise ValueError("Strike (K) must be provided for fixed strike lookback options.")
        if 'floating_strike' in lookback_type and K is not None:
            warnings.warn("Strike (K) is not used for floating strike lookback options. It will be ignored.")

        for i in range(num_paths):
            current_S = self.S0
            min_S_path = self.S0
            max_S_path = self.S0

            for j in range(num_steps):
                t_remaining = T - (j * dt)
                t_remaining = np.maximum(t_remaining, 1e-10) # Ensure T is positive for vol lookup

                sigma_inst = np.nan
                # Fetch instantaneous volatility from the chosen surface
                if model_type == 'LocalVol':
                    if self.local_vol_surface:
                        sigma_inst = self.get_local_volatility(current_S, t_remaining)
                elif model_type == 'SSVI':
                    if self.global_ssvi_params is not None:
                        sigma_inst = self.ssvi_implied_volatility(current_S, t_remaining, self.global_ssvi_params, S0_override=current_S)
                elif model_type == 'SABR':
                    if self.sabr_params_per_maturity:
                        sigma_inst = self.get_model_implied_volatility(current_S, t_remaining, model_type='SABR', S0_override=current_S, r_override=self.r, q_override=self.q)

                if np.isnan(sigma_inst) or sigma_inst <= 1e-8:
                    sigma_inst = 0.2 # Fallback

                z = np.random.normal(0, 1)

                next_S = current_S * np.exp((self.r - self.q - 0.5 * sigma_inst**2) * dt + sigma_inst * np.sqrt(dt) * z)
                next_S = np.maximum(next_S, 1e-6)

                # Update min/max for the path
                min_S_path = np.minimum(min_S_path, next_S)
                max_S_path = np.maximum(max_S_path, next_S)

                current_S = next_S

            terminal_S = current_S
            final_payoff = 0.0

            # Calculate payoff based on lookback type
            if lookback_type == 'fixed_strike_call':
                final_payoff = np.maximum(max_S_path - K, 0)
            elif lookback_type == 'fixed_strike_put':
                final_payoff = np.maximum(K - min_S_path, 0)
            elif lookback_type == 'floating_strike_call': # Payoff is S_T - S_min
                final_payoff = np.maximum(terminal_S - min_S_path, 0)
            elif lookback_type == 'floating_strike_put': # Payoff is S_max - S_T
                final_payoff = np.maximum(max_S_path - terminal_S, 0)

            payoffs[i] = final_payoff

        estimated_price = np.mean(payoffs) * np.exp(-self.r * T)
        print(f"  Monte Carlo Lookback Option Price estimated: {estimated_price:.4f}")
        return estimated_price


    # --- Visualization ---
    def plot_calibration_results(self, model_type: str = 'SSVI'):
        """
        Plots the market implied volatility smile against the calibrated model's
        implied volatility smile for each fetched maturity.
        Generates a separate plot for each maturity for clear visualization.

        Args:
            model_type (str, optional): 'SSVI', 'LocalVol', or 'SABR'. Specifies which model's
                                        calibration results to plot. Defaults to 'SSVI'.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available to plot calibration results.")
            return

        if (model_type == 'SSVI' and self.global_ssvi_params is None) or \
           (model_type == 'LocalVol' and not self.poly_coeffs_implied_vol) or \
           (model_type == 'SABR' and not self.sabr_params_per_maturity):
            print(f"No {model_type} model calibrated or hardcoded to plot its implied smile.")
            return


        print(f"\nPlotting {model_type} Implied Volatility Smile calibration results...")

        plot_data = self.market_option_data.dropna(subset=['implied_vol']).copy()

        for exp_str, T_val in sorted(self.fetched_expirations, key=lambda x: x[1]):
            maturity_data = plot_data[plot_data['expiration_date'] == exp_str].sort_values(by='strike')

            if maturity_data.empty:
                continue

            strikes = maturity_data['strike'].values
            option_types = maturity_data['option_type'].tolist()
            market_implied_vols = maturity_data['implied_vol'].values

            min_strike_ratio = np.min(strikes) / self.S0 if len(strikes) > 0 else 0.8
            max_strike_ratio = np.max(strikes) / self.S0 if len(strikes) > 0 else 1.2

            strike_range_plot = np.linspace(min_strike_ratio * self.S0 * 0.9, max_strike_ratio * self.S0 * 1.1, 100)

            model_vols_curve = []
            if model_type == 'SSVI':
                model_vols_curve = [self.ssvi_implied_volatility(s, T_val, self.global_ssvi_params) for s in strike_range_plot]
            elif model_type == 'LocalVol':
                model_vols_curve = [self.get_implied_vol_from_poly(s, T_val) for s in strike_range_plot]
            elif model_type == 'SABR':
                model_vols_curve = [self.get_model_implied_volatility(s, T_val, model_type='SABR') for s in strike_range_plot]

            model_vols_curve = np.array(model_vols_curve)

            calls_mask = np.array(option_types) == 'call'
            puts_mask = np.array(option_types) == 'put'

            plt.figure(figsize=(10, 6))
            plt.scatter(strikes[calls_mask], market_implied_vols[calls_mask], color='blue', label='Market Implied Vol (Calls)', marker='o', alpha=0.6)
            plt.scatter(strikes[puts_mask], market_implied_vols[puts_mask], color='green', label='Market Implied Vol (Puts)', marker='o', alpha=0.6)

            plt.plot(strike_range_plot, model_vols_curve, color='red', linestyle='-', linewidth=2, label=f'Calibrated {model_type} Volatility Smile')

            plt.title(f'{model_type} Implied Volatility Smile for {self.ticker_symbol} (T={T_val:.2f} years, Exp: {exp_str})')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_local_volatility_surface(self):
        """
        Plots the derived 3D Local Volatility surface as a heatmap or contour plot.
        """
        if not self.local_vol_surface:
            print("Local Volatility surface not derived. Cannot plot.")
            return

        print("\nPlotting 3D Local Volatility Surface (Heatmap)...")

        times_grid = sorted(list(self.local_vol_surface.keys()))
        strikes_grid = sorted(list(self.local_vol_surface[times_grid[0]].keys()))

        # Create a 2D array for the local volatility values
        local_vols_array = np.zeros((len(times_grid), len(strikes_grid)))
        for i, T_val in enumerate(times_grid):
            for j, K_val in enumerate(strikes_grid):
                local_vols_array[i, j] = self.local_vol_surface[T_val].get(K_val, np.nan)

        # Create contour plot or heatmap
        plt.figure(figsize=(12, 8))
        X, Y = np.meshgrid(strikes_grid, times_grid)

        # Use contourf for a filled contour plot
        contour = plt.contourf(X, Y, local_vols_array, levels=20, cmap='viridis', alpha=0.9)
        plt.colorbar(contour, label='Local Volatility')
        plt.contour(X, Y, local_vols_array, levels=20, colors='black', linewidths=0.5) # Add black lines for contours

        plt.title(f'Derived Local Volatility Surface for {self.ticker_symbol}')
        plt.xlabel('Strike Price (K)')
        plt.ylabel('Time to Maturity (T)')
        plt.grid(True)
        plt.show()


    def plot_greeks_for_single_option(self, option_data_row: pd.Series, model_type: str = 'SSVI'):
        """
        Plots Delta and Gamma as a function of strike for a specific option,
        displaying how these Greeks vary across the smile for that maturity,
        using either the SSVI, Local Vol, or SABR model.

        Args:
            option_data_row (pd.Series): A single row from the market_option_data DataFrame.
            model_type (str, optional): 'SSVI', 'LocalVol', or 'SABR'. Defaults to 'SSVI'.
        """
        print(f"\nPlotting Greeks for single example option ({model_type} Model - K={option_data_row['strike']:.2f}, T={option_data_row['T']:.4f}, Type={option_data_row['option_type']})...")

        K_val = option_data_row['strike']
        T_val = option_data_row['T']
        option_type = option_data_row['option_type']

        if (model_type == 'SSVI' and self.global_ssvi_params is None) or \
           (model_type == 'LocalVol' and not self.local_vol_surface) or \
           (model_type == 'SABR' and (T_val not in self.sabr_params_per_maturity or np.any(np.isnan(self.sabr_params_per_maturity.get(T_val, []))))):
            print(f"Skipping plot for T={T_val:.4f}: {model_type} model not calibrated/derived or data unavailable.")
            return

        # Define plotting range for strikes around the ATM price (S0)
        # This ensures the plot covers a relevant range around the money-ness of options
        strike_min_plot = self.S0 * 0.8
        strike_max_plot = self.S0 * 1.2
        strikes_for_plot = np.linspace(strike_min_plot, strike_max_plot, 50)

        deltas = []
        gammas = []
        for K in strikes_for_plot:
            greeks = self.calculate_greeks(self.S0, K, T_val, self.r, self.q, option_type, model_type=model_type)
            deltas.append(greeks.get('Delta'))
            gammas.append(greeks.get('Gamma'))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(strikes_for_plot, deltas, label=f'Delta ({model_type})')
        plt.title(f'Delta vs. Strike ({model_type} Model, T={T_val:.2f} years, {self.ticker_symbol})')
        plt.xlabel('Strike Price')
        plt.ylabel('Delta')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(strikes_for_plot, gammas, label=f'Gamma ({model_type})', color='orange')
        plt.title(f'Gamma vs. Strike ({model_type} Model, T={T_val:.2f} years, {self.ticker_symbol})')
        plt.xlabel('Strike Price')
        plt.ylabel('Gamma')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_greeks_across_surface(self, model_type: str = 'SSVI'):
        """
        Generates and plots Delta, Gamma, Vega, Theta, and Rho
        as functions of strike for *each fetched maturity*, using the specified model.

        Args:
            model_type (str, optional): 'SSVI', 'LocalVol', or 'SABR'. Defaults to 'SSVI'.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available to plot Greeks across surface.")
            return

        if (model_type == 'SSVI' and self.global_ssvi_params is None) or \
           (model_type == 'LocalVol' and not self.local_vol_surface) or \
           (model_type == 'SABR' and not self.sabr_params_per_maturity):
            print(f"No {model_type} model calibrated/derived or data unavailable to plot Greeks across surface.")
            return

        print(f"\nPlotting Greeks across the {model_type} Implied Volatility Surface...")

        greek_types = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        min_strike_for_plot = self.S0 * 0.75
        max_strike_for_plot = self.S0 * 1.25
        strikes_for_plot = np.linspace(min_strike_for_plot, max_strike_for_plot, 100)

        for greek_type in greek_types:
            plt.figure(figsize=(10, 6))

            for exp_str, T_val in sorted(self.fetched_expirations, key=lambda x: x[1]):
                # Skip if SABR params for this T are NaN (failed calibration)
                if model_type == 'SABR' and (T_val not in self.sabr_params_per_maturity or np.any(np.isnan(self.sabr_params_per_maturity.get(T_val, [])))):
                    continue

                greek_values_for_maturity = []
                for K in strikes_for_plot:
                    greeks_calc = self.calculate_greeks(self.S0, K, T_val, self.r, self.q, option_type='call', model_type=model_type)
                    greek_values_for_maturity.append(greeks_calc.get(greek_type))

                plt.plot(strikes_for_plot, greek_values_for_maturity, label=f'T={T_val:.2f} yrs')

            plt.title(f'{greek_type} vs. Strike for {self.ticker_symbol} ({model_type} Model)')
            plt.xlabel('Strike Price')
            plt.ylabel(greek_type)
            plt.legend(title='Time to Maturity', loc='best')
            plt.grid(True)
            plt.show()


    def plot_all_model_calibration_comparison(self, T_val: float):
        """
        Plots the market implied volatility smile against the calibrated SSVI,
        Local Volatility (from polynomial fit), and SABR model smiles for a specific maturity T.
        All curves are shown on a single graph for direct comparison.

        Args:
            T_val (float): The time to maturity for which to plot the comparison.
        """
        if self.market_option_data is None or self.market_option_data.empty:
            print("No market option data available to plot calibration results.")
            return

        # Check if all models are calibrated/derived for this T_val
        ssvi_available = self.global_ssvi_params is not None
        local_vol_available = T_val in self.poly_coeffs_implied_vol and not np.any(np.isnan(self.poly_coeffs_implied_vol.get(T_val, [])))
        sabr_available = T_val in self.sabr_params_per_maturity and not np.any(np.isnan(self.sabr_params_per_maturity.get(T_val, [])))

        if not (ssvi_available or local_vol_available or sabr_available):
            print(f"No valid model calibrations found for T={T_val:.4f} to plot comparison.")
            return

        exp_str_for_title = next((exp[0] for exp in self.fetched_expirations if exp[1] == T_val), f"T={T_val:.4f}")

        print(f"\nPlotting Comparison of All Model Implied Volatility Smiles for T={T_val:.4f}...")

        plot_data = self.market_option_data[self.market_option_data['T'] == T_val].dropna(subset=['implied_vol']).copy().sort_values(by='strike')

        if plot_data.empty:
            print(f"No valid market implied volatilities for T={T_val:.4f} to compare models.")
            return

        strikes_market = plot_data['strike'].values
        market_implied_vols = plot_data['implied_vol'].values
        option_types = plot_data['option_type'].tolist()

        min_strike_ratio = np.min(strikes_market) / self.S0 if len(strikes_market) > 0 else 0.8
        max_strike_ratio = np.max(strikes_market) / self.S0 if len(strikes_market) > 0 else 1.2

        strike_range_plot = np.linspace(min_strike_ratio * self.S0 * 0.9, max_strike_ratio * self.S0 * 1.1, 100)

        plt.figure(figsize=(12, 7))

        # Plot Market Data
        calls_mask = np.array(option_types) == 'call'
        puts_mask = np.array(option_types) == 'put'
        plt.scatter(strikes_market[calls_mask], market_implied_vols[calls_mask], color='blue', label='Market Implied Vol (Calls)', marker='o', alpha=0.6, s=50)
        plt.scatter(strikes_market[puts_mask], market_implied_vols[puts_mask], color='green', label='Market Implied Vol (Puts)', marker='o', alpha=0.6, s=50)

        # Plot Model Curves
        if ssvi_available:
            ssvi_vols = [self.ssvi_implied_volatility(s, T_val, self.global_ssvi_params) for s in strike_range_plot]
            plt.plot(strike_range_plot, ssvi_vols, color='red', linestyle='--', linewidth=2, label='SSVI Model Implied Vol')
        else:
            print(f"  SSVI model not available for T={T_val:.4f} comparison plot.")

        if local_vol_available:
            local_vols_from_poly = [self.get_implied_vol_from_poly(s, T_val) for s in strike_range_plot]
            plt.plot(strike_range_plot, local_vols_from_poly, color='purple', linestyle='-.', linewidth=2, label='Local Vol (Poly Fit) Implied Vol')
        else:
            print(f"  Local Vol model (polynomial fit) not available for T={T_val:.4f} comparison plot.")

        if sabr_available:
            sabr_vols = [self.get_model_implied_volatility(s, T_val, model_type='SABR') for s in strike_range_plot]
            plt.plot(strike_range_plot, sabr_vols, color='orange', linestyle=':', linewidth=2, label='SABR Model Implied Vol')
        else:
            print(f"  SABR model not available for T={T_val:.4f} comparison plot.")

        plt.title(f'Implied Volatility Model Comparison for {self.ticker_symbol} (T={T_val:.2f} years, Exp: {exp_str_for_title})')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- Main Execution Block (for testing and demonstration) ---
if __name__ == "__main__":
    print("Starting Hybrid Volatility Surface Model (SSVI & Local Volatility & SABR) development...")

    RUN_CALIBRATION = True
    RUN_SSVI_MODEL = True # Control flag to run SSVI calibration and plots
    RUN_LOCAL_VOL_MODEL = True # Control flag to run Local Volatility derivation and plots
    RUN_SABR_MODEL = True # Control flag to run SABR calibration and plots (NEW)
    RUN_MONTE_CARLO_ASIAN_EXAMPLE = True # Control flag to run Monte Carlo Asian Option pricing
    RUN_MONTE_CARLO_BARRIER_EXAMPLE = True # Control flag to run Monte Carlo Barrier Option pricing (NEW)
    RUN_MONTE_CARLO_LOOKBACK_EXAMPLE = True # Control flag to run Monte Carlo Lookback Option pricing (NEW)
    RUN_ALL_MODEL_COMPARISON_PLOT = True # New flag to run the combined comparison plot


    S0_initial = 100.0
    r_val = 0.01
    q_val = 0.0

    # Initialize the Hybrid model
    hybrid_model = HybridVolatilitySurfaceModel(S0=S0_initial, r=r_val, q=q_val)

    ticker_symbol = 'MSFT'
    hybrid_model.ticker_symbol = ticker_symbol

    historical_data = hybrid_model.fetch_stock_data(ticker_symbol)
    if historical_data is None:
        print(f"Failed to fetch stock data for {ticker_symbol}. Cannot proceed with analysis.")
        exit()

    market_option_data = hybrid_model.fetch_option_data(ticker_symbol, num_expirations=5)
    print(f"Number of valid expirations found: {len(hybrid_model.fetched_expirations)}")
    if market_option_data is None:
        print(f"Failed to fetch valid option data for {ticker_symbol}. Cannot proceed with analysis.")
        print("Please check the ticker symbol and available expiration dates (e.g., using `yf.Ticker('MSFT').options`).")
        exit()

    # --- Calibrate All Models (or selected ones) ---
    if RUN_CALIBRATION:
        hybrid_model.calibrate_all_models(
            run_ssvi=RUN_SSVI_MODEL,
            run_local_vol=RUN_LOCAL_VOL_MODEL,
            run_sabr=RUN_SABR_MODEL, # New SABR flag
            num_generations=200, # Generations for SSVI
            poly_degree=3        # Degree for Local Vol polynomial fit
        )
    else:
        print("\n--- SKIPPING CALIBRATION ---")
        # Hardcoded parameters for demonstration if calibration is skipped
        if RUN_SSVI_MODEL:
            print("Using hardcoded global SSVI parameters.")
            hybrid_model.global_ssvi_params = [0.03, -0.05, 0.02, -0.7, 0.3, 0.5]
            print(f"Hardcoded Global SSVI Parameters: {hybrid_model.global_ssvi_params}")
        if RUN_LOCAL_VOL_MODEL:
            print("Using hardcoded polynomial coefficients for Local Vol.")
            hybrid_model.poly_coeffs_implied_vol = {
                0.2683: np.array([0.05, -0.1, 0.2, 0.25]), # Example for T1
                0.3450: np.array([0.06, -0.08, 0.18, 0.24]), # Example for T2
                0.5175: np.array([0.07, -0.07, 0.15, 0.23])  # Example for T3
            }
            hybrid_model.derive_local_volatility_surface()
            print(f"Hardcoded Local Vol Polynomial Coeffs: {hybrid_model.poly_coeffs_implied_vol}")
        if RUN_SABR_MODEL:
            print("Using hardcoded SABR parameters per maturity.")
            hybrid_model.sabr_params_per_maturity = {
                0.2683: [0.25, 0.5, -0.3, 0.4], # alpha, beta, rho, nu
                0.3450: [0.22, 0.5, -0.2, 0.35],
                0.5175: [0.20, 0.5, -0.1, 0.3]
            }
            print(f"Hardcoded SABR Params: {hybrid_model.sabr_params_per_maturity}")


    print("\n--- Post-Analysis and Pricing (using calibrated/hardcoded model parameters) ---")

    print("Current model base parameters:")
    print(f"  S0: {hybrid_model.S0:.2f}, r: {hybrid_model.r:.4f}, q: {hybrid_model.q:.4f}")

    # Perform example pricing and Greek calculation for all models
    if hybrid_model.market_option_data is not None and not hybrid_model.market_option_data.empty:
        # Find the first available option for demonstration
        example_option = hybrid_model.market_option_data.iloc[0]

        strike_example = example_option['strike']
        option_type_example = example_option['option_type']
        T_example = example_option['T']
        market_implied_vol_example = example_option['implied_vol']

        # --- SSVI Model Example ---
        if RUN_SSVI_MODEL and hybrid_model.global_ssvi_params is not None:
            model_ssvi_vol_example = hybrid_model.get_model_implied_volatility(
                strike_example, T_example, model_type='SSVI'
            )
            model_price_bs_from_ssvi = hybrid_model.black_scholes_price(
                S=hybrid_model.S0, K=strike_example, T=T_example, r=hybrid_model.r, q=hybrid_model.q,
                sigma_val=model_ssvi_vol_example, option_type=option_type_example
            )
            print(f"\n--- SSVI Model Results for Example Option ({option_type_example.upper()}, K={strike_example:.2f}, T={T_example:.4f}) ---")
            print(f"  Market Price: {example_option['market_price']:.4f}")
            print(f"  Market Implied Vol: {market_implied_vol_example:.4f}")
            print(f"  Model SSVI Implied Vol: {model_ssvi_vol_example:.4f}")
            print(f"  Model Price (BS with Model SSVI Vol): {model_price_bs_from_ssvi:.4f}")
            print(f"  Difference (Model SSVI IV - Market IV): {model_ssvi_vol_example - market_implied_vol_example:.4f}")
            print(f"  Difference (Model Price - Market Price): {model_price_bs_from_ssvi - example_option['market_price']:.4f}")

            print("\n--- Greeks Calculation for Example Option (SSVI Model) ---")
            greeks_ssvi_example = hybrid_model.calculate_greeks(
                hybrid_model.S0, strike_example, T_example, hybrid_model.r, hybrid_model.q, option_type_example, model_type='SSVI'
            )
            for greek, value in greeks_ssvi_example.items():
                print(f"  {greek}: {value:.4f}")
            hybrid_model.plot_greeks_for_single_option(example_option, model_type='SSVI')

        # --- Local Volatility Model Example ---
        if RUN_LOCAL_VOL_MODEL and hybrid_model.local_vol_surface:
            model_local_vol_implied_example = hybrid_model.get_model_implied_volatility(
                strike_example, T_example, model_type='LocalVol'
            )
            model_price_bs_from_local_vol = hybrid_model.black_scholes_price(
                S=hybrid_model.S0, K=strike_example, T=T_example, r=hybrid_model.r, q=hybrid_model.q,
                sigma_val=model_local_vol_implied_example, option_type=option_type_example
            )
            print(f"\n--- Local Volatility Model Results for Example Option ({option_type_example.upper()}, K={strike_example:.2f}, T={T_example:.4f}) ---")
            print(f"  Market Price: {example_option['market_price']:.4f}")
            print(f"  Market Implied Vol: {market_implied_vol_example:.4f}")
            print(f"  Model Local Vol Implied Vol: {model_local_vol_implied_example:.4f}")
            print(f"  Model Price (BS with Model Local Vol IV): {model_price_bs_from_local_vol:.4f}")
            print(f"  Difference (Model Local Vol IV - Market IV): {model_local_vol_implied_example - market_implied_vol_example:.4f}")
            print(f"  Difference (Model Price - Market Price): {model_price_bs_from_local_vol - example_option['market_price']:.4f}")

            print("\n--- Greeks Calculation for Example Option (Local Volatility Model) ---")
            greeks_local_vol_example = hybrid_model.calculate_greeks(
                hybrid_model.S0, strike_example, T_example, hybrid_model.r, hybrid_model.q, option_type_example, model_type='LocalVol'
            )
            for greek, value in greeks_local_vol_example.items():
                print(f"  {greek}: {value:.4f}")
            hybrid_model.plot_greeks_for_single_option(example_option, model_type='LocalVol')

        # --- SABR Model Example ---
        if RUN_SABR_MODEL and T_example in hybrid_model.sabr_params_per_maturity and not np.any(np.isnan(hybrid_model.sabr_params_per_maturity.get(T_example, []))):
            model_sabr_vol_example = hybrid_model.get_model_implied_volatility(
                strike_example, T_example, model_type='SABR'
            )
            model_price_bs_from_sabr = hybrid_model.black_scholes_price(
                S=hybrid_model.S0, K=strike_example, T=T_example, r=hybrid_model.r, q=hybrid_model.q,
                sigma_val=model_sabr_vol_example, option_type=option_type_example
            )
            print(f"\n--- SABR Model Results for Example Option ({option_type_example.upper()}, K={strike_example:.2f}, T={T_example:.4f}) ---")
            print(f"  Market Price: {example_option['market_price']:.4f}")
            print(f"  Market Implied Vol: {market_implied_vol_example:.4f}")
            print(f"  Model SABR Implied Vol: {model_sabr_vol_example:.4f}")
            print(f"  Model Price (BS with Model SABR Vol): {model_price_bs_from_sabr:.4f}")
            print(f"  Difference (Model SABR IV - Market IV): {model_sabr_vol_example - market_implied_vol_example:.4f}")
            print(f"  Difference (Model Price - Market Price): {model_price_bs_from_sabr - example_option['market_price']:.4f}")

            print("\n--- Greeks Calculation for Example Option (SABR Model) ---")
            greeks_sabr_example = hybrid_model.calculate_greeks(
                hybrid_model.S0, strike_example, T_example, hybrid_model.r, hybrid_model.q, option_type_example, model_type='SABR'
            )
            for greek, value in greeks_sabr_example.items():
                print(f"  {greek}: {value:.4f}")
            hybrid_model.plot_greeks_for_single_option(example_option, model_type='SABR')

        # --- Monte Carlo Asian Option Example ---
        if RUN_MONTE_CARLO_ASIAN_EXAMPLE:
            print("\n--- Monte Carlo Asian Option Pricing Example ---")
            # Using the first fetched maturity's T as an example
            asian_option_K = hybrid_model.S0 # Example: ATM Asian option
            asian_option_T = hybrid_model.fetched_expirations[0][1] if hybrid_model.fetched_expirations else 0.5

            # Price using Local Volatility for path generation (most theoretically consistent)
            if RUN_LOCAL_VOL_MODEL and hybrid_model.local_vol_surface:
                print(f"Pricing Asian Call (K={asian_option_K:.2f}, T={asian_option_T:.4f}) using Monte Carlo with Local Volatility surface...")
                asian_call_price_lv = hybrid_model.monte_carlo_asian_option_price(
                    K=asian_option_K, T=asian_option_T, option_type='call',
                    model_type='LocalVol', num_paths=20000, num_steps=252 # Daily steps for 1 year
                )
                print(f"  Asian Call Price (Local Vol MC): {asian_call_price_lv:.4f}")
            else:
                print("Skipping Monte Carlo for Asian Option (Local Vol) as local volatility surface not derived.")

            # Price using SSVI for path generation (as fallback/alternative)
            if RUN_SSVI_MODEL and hybrid_model.global_ssvi_params is not None:
                print(f"Pricing Asian Call (K={asian_option_K:.2f}, T={asian_option_T:.4f}) using Monte Carlo with SSVI model...")
                asian_call_price_ssvi = hybrid_model.monte_carlo_asian_option_price(
                    K=asian_option_K, T=asian_option_T, option_type='call',
                    model_type='SSVI', num_paths=20000, num_steps=252
                )
                print(f"  Asian Call Price (SSVI MC): {asian_call_price_ssvi:.4f}")
            else:
                print("Skipping Monte Carlo for Asian Option (SSVI) as SSVI model not calibrated.")

            # Price using SABR for path generation (as fallback/alternative)
            if RUN_SABR_MODEL and hybrid_model.sabr_params_per_maturity:
                print(f"Pricing Asian Call (K={asian_option_K:.2f}, T={asian_option_T:.4f}) using Monte Carlo with SABR model...")
                asian_call_price_sabr = hybrid_model.monte_carlo_asian_option_price(
                    K=asian_option_K, T=asian_option_T, option_type='call',
                    model_type='SABR', num_paths=20000, num_steps=252
                )
                print(f"  Asian Call Price (SABR MC): {asian_call_price_sabr:.4f}")
            else:
                print("Skipping Monte Carlo for Asian Option (SABR) as SABR model not calibrated.")

        # --- Monte Carlo Barrier Option Example (NEW) ---
        if RUN_MONTE_CARLO_BARRIER_EXAMPLE:
            print("\n--- Monte Carlo Barrier Option Pricing Example ---")
            barrier_T_example = hybrid_model.fetched_expirations[0][1] if hybrid_model.fetched_expirations else 0.5
            barrier_K_example = hybrid_model.S0 # ATM strike for simplicity

            # Example 1: Up-and-Out Call option using Local Volatility
            up_out_barrier = hybrid_model.S0 * 1.15 # 15% above current price
            if RUN_LOCAL_VOL_MODEL and hybrid_model.local_vol_surface:
                print(f"Pricing Up-and-Out Call (K={barrier_K_example:.2f}, T={barrier_T_example:.4f}, Barrier={up_out_barrier:.2f}) using Monte Carlo with Local Volatility surface...")
                uo_call_price_lv = hybrid_model.monte_carlo_barrier_option_price(
                    K=barrier_K_example, T=barrier_T_example, barrier=up_out_barrier,
                    barrier_type='up_and_out_call', option_type='call', model_type='LocalVol',
                    num_paths=20000, num_steps=252
                )
                print(f"  Up-and-Out Call Price (Local Vol MC): {uo_call_price_lv:.4f}")
            else:
                print("Skipping Monte Carlo for Up-and-Out Call (Local Vol) as local volatility surface not derived.")

            # Example 2: Down-and-In Put option using SSVI
            down_in_barrier = hybrid_model.S0 * 0.85 # 15% below current price
            if RUN_SSVI_MODEL and hybrid_model.global_ssvi_params is not None:
                print(f"Pricing Down-and-In Put (K={barrier_K_example:.2f}, T={barrier_T_example:.4f}, Barrier={down_in_barrier:.2f}) using Monte Carlo with SSVI model...")
                di_put_price_ssvi = hybrid_model.monte_carlo_barrier_option_price(
                    K=barrier_K_example, T=barrier_T_example, barrier=down_in_barrier,
                    barrier_type='down_and_in_put', option_type='put', model_type='SSVI',
                    num_paths=20000, num_steps=252
                )
                print(f"  Down-and-In Put Price (SSVI MC): {di_put_price_ssvi:.4f}")
            else:
                print("Skipping Monte Carlo for Down-and-In Put (SSVI) as SSVI model not calibrated.")

            # Example 3: Up-and-In Call option using SABR (New addition)
            up_in_barrier_sabr = hybrid_model.S0 * 1.10 # 10% above current price
            if RUN_SABR_MODEL and hybrid_model.sabr_params_per_maturity:
                print(f"Pricing Up-and-In Call (K={barrier_K_example:.2f}, T={barrier_T_example:.4f}, Barrier={up_in_barrier_sabr:.2f}) using Monte Carlo with SABR model...")
                ui_call_price_sabr = hybrid_model.monte_carlo_barrier_option_price(
                    K=barrier_K_example, T=barrier_T_example, barrier=up_in_barrier_sabr,
                    barrier_type='up_and_in_call', option_type='call', model_type='SABR',
                    num_paths=20000, num_steps=252
                )
                print(f"  Up-and-In Call Price (SABR MC): {ui_call_price_sabr:.4f}")
            else:
                print("Skipping Monte Carlo for Up-and-In Call (SABR) as SABR model not calibrated.")

        # --- Monte Carlo Lookback Option Example (NEW) ---
        if RUN_MONTE_CARLO_LOOKBACK_EXAMPLE:
            print("\n--- Monte Carlo Lookback Option Pricing Example ---")
            lookback_T_example = hybrid_model.fetched_expirations[0][1] if hybrid_model.fetched_expirations else 0.5
            lookback_K_example = hybrid_model.S0 # Use ATM strike for fixed strike example

            # Example 1: Floating Strike Lookback Call using Local Volatility
            if RUN_LOCAL_VOL_MODEL and hybrid_model.local_vol_surface:
                print(f"Pricing Floating Strike Lookback Call (T={lookback_T_example:.4f}) using Monte Carlo with Local Volatility surface...")
                float_lb_call_price_lv = hybrid_model.monte_carlo_lookback_option_price(
                    K=None, T=lookback_T_example, # K is None for floating strike
                    lookback_type='floating_strike_call', option_type='call', model_type='LocalVol',
                    num_paths=20000, num_steps=252
                )
                print(f"  Floating Strike Lookback Call Price (Local Vol MC): {float_lb_call_price_lv:.4f}")
            else:
                print("Skipping Monte Carlo for Floating Strike Lookback Call (Local Vol) as local volatility surface not derived.")

            # Example 2: Fixed Strike Lookback Put using SABR
            if RUN_SABR_MODEL and hybrid_model.sabr_params_per_maturity:
                print(f"Pricing Fixed Strike Lookback Put (K={lookback_K_example:.2f}, T={lookback_T_example:.4f}) using Monte Carlo with SABR model...")
                fixed_lb_put_price_sabr = hybrid_model.monte_carlo_lookback_option_price(
                    K=lookback_K_example, T=lookback_T_example, # K is required for fixed strike
                    lookback_type='fixed_strike_put', option_type='put', model_type='SABR',
                    num_paths=20000, num_steps=252
                )
                print(f"  Fixed Strike Lookback Put Price (SABR MC): {fixed_lb_put_price_sabr:.4f}")
            else:
                print("Skipping Monte Carlo for Fixed Strike Lookback Put (SABR) as SABR model not calibrated.")

            # Example 3: Floating Strike Lookback Put using SSVI (New addition)
            if RUN_SSVI_MODEL and hybrid_model.global_ssvi_params is not None:
                print(f"Pricing Floating Strike Lookback Put (T={lookback_T_example:.4f}) using Monte Carlo with SSVI model...")
                float_lb_put_price_ssvi = hybrid_model.monte_carlo_lookback_option_price(
                    K=None, T=lookback_T_example,
                    lookback_type='floating_strike_put', option_type='put', model_type='SSVI',
                    num_paths=20000, num_steps=252
                )
                print(f"  Floating Strike Lookback Put Price (SSVI MC): {float_lb_put_price_ssvi:.4f}")
            else:
                print("Skipping Monte Carlo for Floating Strike Lookback Put (SSVI) as SSVI model not calibrated.")


    # --- Plotting All Calibration Results and Greeks ---
    if RUN_SSVI_MODEL and hybrid_model.global_ssvi_params is not None:
        hybrid_model.plot_calibration_results(model_type='SSVI')
        hybrid_model.plot_greeks_across_surface(model_type='SSVI')
    else:
        print("\nSkipping SSVI plots as no valid global SSVI parameters were calibrated or hardcoded.")

    if RUN_LOCAL_VOL_MODEL and hybrid_model.local_vol_surface:
        hybrid_model.plot_calibration_results(model_type='LocalVol') # This plots implied vol from polynomial fit
        hybrid_model.plot_local_volatility_surface() # This plots the derived local vol surface itself
        hybrid_model.plot_greeks_across_surface(model_type='LocalVol')
    else:
        print("\nSkipping Local Volatility plots as surface was not derived or hardcoded.")

    if RUN_SABR_MODEL and hybrid_model.sabr_params_per_maturity:
        hybrid_model.plot_calibration_results(model_type='SABR')
        hybrid_model.plot_greeks_across_surface(model_type='SABR')
    else:
        print("\nSkipping SABR plots as no valid SABR parameters were calibrated or hardcoded for any maturity.")

    # --- New: Plot All Model Calibration Comparison ---
    if RUN_ALL_MODEL_COMPARISON_PLOT and hybrid_model.fetched_expirations:
        # Use the first fetched maturity for the comparison plot
        comparison_T_val = hybrid_model.fetched_expirations[0][1]
        hybrid_model.plot_all_model_calibration_comparison(comparison_T_val)
    elif RUN_ALL_MODEL_COMPARISON_PLOT:
        print("\nSkipping all-model comparison plot as no valid maturities were fetched or models calibrated.")


    print("\n--- Hybrid Volatility Surface Model Development Progress ---")
    print("- Fetched real-world stock data using yfinance. (Done)")
    print("- Implemented Black-Scholes option pricing and Implied Volatility calculation. (Done)")
    print("- Implemented the SSVI (Smoothed SVI) global parameterization and calibration. (Done)")
    print("- Added comprehensive static and calendar spread arbitrage-free penalties to SSVI. (Done)")
    print("- Implemented Local Volatility Surface Derivation (Polynomial fitting & Dupire's formula). (Done)")
    print("- Integrated SABR Implied Volatility model and per-maturity calibration. (Done)")
    print("- Updated pricing, Greeks, and plotting functions for SSVI, Local Vol, and SABR. (Done)")
    print("- Implemented Monte Carlo simulation for pricing Arithmetic Mean Asian Options. (Done)")
    print("- Implemented Monte Carlo simulation for pricing Barrier Options (Up/Down, In/Out). (Done)")
    print("- Implemented Monte Carlo simulation for pricing Lookback Options (Fixed/Floating Strike). (Done)")
    print("- Integrated volatility from the chosen surface (SSVI, Local Vol, or SABR) into all Monte Carlo path generation. (Done)")
    print("- Added extensive documentation and comments throughout the codebase. (Done)")
    print("- **Added combined plot for comparing all model implied volatility fits to market data.** (Done)")
    
