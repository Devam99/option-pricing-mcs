"""
option.py - Classes and functions for option pricing using Monte Carlo simulation.

This module provides classes and functions to model and price options using Monte Carlo methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional

class OptionType(Enum):
    """Enumeration for option types."""
    CALL = "call"
    PUT = "put"

@dataclass
class Option(ABC):
    """
    Abstract base class for options
    Attributes:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (OptionType): Type of the option (CALL or PUT)
    """
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, option_type: OptionType):
        """
        Initialize the option with given parameters.

        Args:
            S0 (float): Initial stock price.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            option_type (OptionType): Type of the option (CALL or PUT).
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

    def _validate(self):
        """Validate the option parameters."""
        if self.S0 <= 0:
            raise ValueError("Initial stock price (S0) must be greater than 0.")
        if self.K <= 0:
            raise ValueError("Strike price (K) must be greater than 0.")
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be greater than 0.")
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be greater than 0.")


    @abstractmethod
    def payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff of the option at maturity.

        Args:
            S (np.ndarray): Array of stock prices at maturity.

        Returns:
            np.ndarray: Payoff of the option.
        """
        pass

    @property
    def is_call(self) -> bool:
        """Check if the option is a call option."""
        return self.option_type == OptionType.CALL

@dataclass
class EuropeanOption(Option):
    """Class for European options, which can only be exercised at maturity."""
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType

    def __post_init__(self):
        super().__init__(self.S0, self.K, self.T, self.r, self.sigma, self.option_type)


    def payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff of a European option at maturity.

        Args:
            S (np.ndarray): Array of stock prices at maturity.

        Returns:
            np.ndarray: Payoff of the European option.
        """
        if self.is_call:
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def black_scholes_price(self) -> float:
        from scipy.stats import norm

        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.is_call:
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

        return price


@dataclass
class AsianOption(Option):
    """Class for Asian options, which are options where the payoff depends on the average price of the underlying asset over a certain period."""
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType
    n_averaging_points: int = 252  # Default to daily averaging for one year

    def __post_init__(self):
        """Validate the Asian option parameters."""
        super().__init__(self.S0, self.K, self.T, self.r, self.sigma, self.option_type)
        if self.n_averaging_points <= 0:
            raise ValueError("Number of averaging points must be greater than 0.")

    def payoff(self, S_path: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff of an Asian option at maturity.

        Args:
            S_path (np.ndarray): Array of stock prices at maturity.

        Returns:
            np.ndarray: Payoff of the Asian option.
        """
        # Handle both 1D and 2D arrays for S_path
        if S_path.ndim == 1:
            #Single path case
            S_avg = np.mean(S_path)
            if self.is_call:
                payoff = np.maximum(S_avg - self.K, 0)
            else:
                payoff = np.maximum(self.K - S_avg, 0)

        else:
            #Multiple paths case
            S_avg = np.mean(S_path, axis=1)
            if self.is_call:
                payoff = np.maximum(S_avg - self.K, 0)
            else:
                payoff = np.maximum(self.K - S_avg, 0)

        return payoff




    @property
    def dt(self) -> float:
        """Calculate the time step for averaging."""
        return self.T / self.n_averaging_points


#Factory function to create an option instance
def create_option(option_class: str, **kwargs) -> Option:
    option_classes = {
        "EuropeanOption": EuropeanOption,
        "AsianOption": AsianOption
    }
    if option_class.lower() not in option_classes:
        raise ValueError(f"Unknown option class: {option_class}")

    if "option_type" not in kwargs and isinstance(kwargs["option_type"], str):
        kwargs["option_type"] = OptionType(kwargs["option_type"].lower())

    return option_classes[option_class.lower()](**kwargs)


if __name__ == "__main__":
    # Example usage

    # Create a European call option

    european_call = EuropeanOption(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )

    print(f"European Call Option:")
    print(f"  Initial Stock Price: ${european_call.S0}")
    print(f"  Strike Price: ${european_call.K}")
    print(f"  Time to Maturity: {european_call.T} years")
    print(f"  Risk-free Rate: {european_call.r:.1%}")
    print(f"  Volatility: {european_call.sigma:.1%}")
    print(f"  Black-Scholes Price: ${european_call.black_scholes_price():.2f}")

    #Test the payoff function
    test_prices = np.array([90, 100, 110])
    euro_payoffs = european_call.payoff(test_prices)
    print(f"\nPayoff for European Call at prices {test_prices}: {euro_payoffs}")

    # Create an Asian put option
    asian_put = AsianOption(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.PUT,
        n_averaging_points=252
    )

    print(f"\nAsian Put Option:")
    print(f"  Initial Stock Price: ${asian_put.S0}")
    print(f"  Strike Price: ${asian_put.K}")
    print(f"  Averaging Points: {asian_put.n_averaging_points}")
    print(f"  Time Step: {asian_put.dt:.4f} years")


    #test the payoff function for Asian option
    test_path = np.array([100, 95, 105, 90, 110])  # Example path
    asian_payoff = asian_put.payoff(test_path)
    print(f"  Path: {test_path}")
    print(f"  Average: {np.mean(test_path):.2f}")
    print(f"  Payoff: {asian_payoff:.2f}")

    # Test the payoff function for Asian option with multiple paths
    test_paths = np.array([[100, 95, 105, 90, 110],
                           [100, 105, 95, 100, 100],
                           [100, 110, 120, 115, 130]])  # Multiple paths

    asian_payoffs = asian_put.payoff(test_paths)
    print(f"\nMultiple paths payoffs: {asian_payoffs}")

    # Test factory function
    euro_put = create_option(
        'EuropeanOption',
        S0=100,
        K=110,
        T=0.5,
        r=0.05,
        sigma=0.3,
        option_type='put'
    )
    print(f"\nFactory-created European Put:")
    print(f"  Black-Scholes Price: ${euro_put.black_scholes_price():.2f}")