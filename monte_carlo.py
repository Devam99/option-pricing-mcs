"""
monte_carlo.py - Monte Carlo simulation for option pricing.

This module provides a class for Monte Carlo simulation to price options.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import time
from option import Option, EuropeanOption, AsianOption

@dataclass
class MonteCarloResult:
    """
    Data class to store the results of the Monte Carlo simulation.

    Attributes:
        price (float): Estimated option price.
        std_error (float): Standard error of the estimate.
        confidence_interval (Tuple[float, float]): Confidence interval for the estimated price.
        computation_time (float): Time taken for the simulation in seconds.
    """
    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    computation_time: float
    m_simulations: int

    def __str__(self):
        return (f" Monte Carlo Result:\n"
                f" Price: ${self.price:.4f}\n"
                f" Standard Error: ${self.std_error:.4f}\n"
                f" 95% Confidence Interval: (%{self.confidence_interval[0]:.4f}, ${self.confidence_interval[1]:.4f})\n"
                f" Computation Time: {self.computation_time:.3f} seconds\n"
                f" Simulations: {self.m_simulations:,}")



class MonteCarloOptionPricer:
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo option pricer.

        Args:
            random_seed (Optional[int]): Seed for the random number generator for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_gbm_paths(self, S0: float, r: float, sigma: float, T: float, n_simulations: int, n_steps: int, antithetic:bool = False) -> np.ndarray:

        dt = T / n_steps

        if antithetic:
            n_anti = n_simulations // 2
            Z = np.random.standard_normal(size=(n_anti, n_steps))
            Z = np.concatenate([Z, -Z], axis=0)
            if n_simulations % 2 != 0:
                Z = np.concatenate([Z, np.random.standard_normal(size=(1, n_steps))], axis=0)

        else:
            Z = np.random.standard_normal((n_simulations, n_steps))


        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S0

        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])

        return paths


    def price_european_option(self, option: EuropeanOption, n_simulations: int = 10000, antithetic: bool = False, control_variable: bool = False) -> MonteCarloResult:
        start_time = time.time()
        dt = option.T

        if antithetic:
            n_anti = n_simulations // 2
            Z = np.random.standard_normal(n_anti)
            Z = np.concatenate([Z, -Z])
            if n_simulations % 2 != 0:
                Z = np.concatenate([Z, np.random.standard_normal(1)])
        else:
            Z = np.random.standard_normal(n_simulations)

        ST = option.S0 * np.exp((option.r - 0.5 * option.sigma ** 2) * option.T + option.sigma * np.sqrt(option.T)* Z)

        payoffs = option.payoff(ST)

        if control_variable and hasattr(option, "black_scholes_price"):
            control_payoffs = ST
            control_mean = option.S0 * np.exp(option.r * option.T)

            cov_matrix = np.cov(payoffs, control_payoffs)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        discounted_payoffs = np.exp(-option.r * option.T) * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        confidence_interval = (price - 1.96 * std_error, price + 1.96 * std_error)
        computation_time = time.time() - start_time

        return MonteCarloResult(price = price,
                                std_error = std_error,
                                confidence_interval = confidence_interval,
                                computation_time = computation_time,
                                m_simulations = n_simulations)

    def price_asian_option(self, option: AsianOption, n_simulations: int = 10000, antithetic: bool = False) -> MonteCarloResult:

        start_time = time.time()

        if antithetic:
            n_anti = n_simulations // 2
            Z = np.random.standard_normal(n_anti)
            Z = np.concatenate([Z, -Z])
            if n_simulations % 2 != 0:
                Z = np.concatenate([Z, np.random.standard_normal(1)])

        else:
            Z = np.random.standard_normal(n_simulations)



