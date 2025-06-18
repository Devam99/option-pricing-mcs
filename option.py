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
    S0: float  # Initial stock price
    K: float # Strike price
    T : float  # Time to maturity in years
    r: float  # Risk-free interest rate
    sigma: float  # Volatility of the underlying asset
    option_type: OptionType

    def __post_init__(self):
        """Validate the option type."""
        if self.SO <= 0:
            raise ValueError("Initial stock price (S0) must be greater than 0.")
        if self.K <= 0:
            raise ValueError("Strike price (K) must be greater than 0.")
        if self.T <= 0:
            raise ValueError("Time to maturity (T) must be greater than 0.")
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be greater than 0.")


    @abstractmethod
    def payoff(self, S: np.ndarry) -> np.ndarray:
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