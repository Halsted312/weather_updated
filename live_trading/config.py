"""
Trading configuration with aggressiveness dial.

The aggressiveness parameter (0-1) controls multiple trading thresholds:
- 0.0 = Conservative (high confidence threshold, low Kelly fraction, long maker timeout)
- 0.5 = Balanced (default)
- 1.0 = Aggressive (low confidence threshold, high Kelly fraction, short maker timeout)
"""

from dataclasses import dataclass, field
from typing import List
import json
from pathlib import Path


@dataclass
class TradingConfig:
    """Master configuration for edge-based live trading."""

    # === AGGRESSIVENESS DIAL (0-1) ===
    aggressiveness: float = 0.5  # 0=conservative, 1=aggressive

    # === POSITION LIMITS ===
    max_bet_per_trade_usd: float = 50.0
    max_daily_loss_usd: float = 600.0
    max_positions_per_city: int = 2
    max_total_positions: int = 12

    # === EDGE CLASSIFIER ===
    edge_confidence_threshold_base: float = 0.5  # Modified by aggressiveness
    edge_threshold_degf: float = 1.5  # Minimum edge in degrees F

    # === ORDER EXECUTION ===
    maker_timeout_base_seconds: int = 120  # 2 minutes base timeout
    volume_timeout_multiplier: float = 1.5  # High volume → longer wait
    volume_lookback_minutes: int = 30  # Volume window for timeout calc

    # === KELLY SIZING ===
    bankroll_usd: float = 10000.0
    kelly_fraction_base: float = 0.25  # Quarter-Kelly base

    # === CITIES ===
    enabled_cities: List[str] = field(default_factory=lambda: ["chicago"])

    # === FEES (Kalshi) ===
    maker_fee_pct: float = 0.0  # Maker orders are free
    taker_fee_pct: float = 0.07  # 7% taker fee

    # === MIN THRESHOLDS ===
    min_ev_per_contract_cents: float = 3.0
    min_bracket_prob: float = 0.35

    # === INFERENCE ===
    inference_cooldown_sec: int = 30  # Cache predictions for 30 seconds

    # === INFERENCE STRATEGY ===
    inference_mode: str = "adaptive"  # 'adaptive', 'classifier_only', 'threshold_only'
    edge_classifier_cities: List[str] = field(default_factory=lambda: ["chicago", "denver"])
    fallback_to_threshold: bool = True  # If classifier missing, use threshold

    # === DERIVED PROPERTIES ===

    @property
    def effective_confidence_threshold(self) -> float:
        """
        Aggressiveness controls edge classifier confidence threshold.

        aggressiveness=0.0 → 0.70 (very conservative, only trade high-confidence edges)
        aggressiveness=0.5 → 0.50 (balanced)
        aggressiveness=1.0 → 0.30 (aggressive, trade more edges)
        """
        return 0.7 - (self.aggressiveness * 0.4)

    @property
    def effective_kelly_fraction(self) -> float:
        """
        Aggressiveness controls Kelly fraction for position sizing.

        aggressiveness=0.0 → 0.10 (1/10 Kelly, very conservative)
        aggressiveness=0.5 → 0.30 (balanced)
        aggressiveness=1.0 → 0.50 (1/2 Kelly, aggressive)
        """
        return 0.1 + (self.aggressiveness * 0.4)

    @property
    def effective_maker_timeout_multiplier(self) -> float:
        """
        Aggressiveness controls how long to wait for maker fills.

        aggressiveness=0.0 → 1.5x base (wait longer for maker fills, minimize fees)
        aggressiveness=0.5 → 1.0x base (balanced)
        aggressiveness=1.0 → 0.5x base (convert to taker faster, ensure execution)
        """
        return 1.5 - (self.aggressiveness * 1.0)

    def compute_maker_timeout_sec(self, recent_volume: int) -> int:
        """
        Compute actual maker timeout based on volume and aggressiveness.

        High volume → wait longer (more likely to fill as maker)
        Low volume → convert to taker faster (ensure execution)

        Args:
            recent_volume: Contracts traded in last volume_lookback_minutes

        Returns:
            Timeout in seconds before converting maker → taker
        """
        # Base timeout modified by aggressiveness
        base = self.maker_timeout_base_seconds * self.effective_maker_timeout_multiplier

        # Volume factor: normalize 0-100 contracts in window to 0-2x multiplier
        volume_factor = min(recent_volume / 100.0, 2.0)

        # High volume → longer timeout (up to 50% longer)
        effective_timeout = base * (1 + volume_factor * 0.5 * self.volume_timeout_multiplier)

        return int(effective_timeout)

    @classmethod
    def from_json(cls, path: Path) -> 'TradingConfig':
        """
        Load config from JSON file, merging with defaults.

        Args:
            path: Path to JSON config file

        Returns:
            TradingConfig instance with values from file + defaults for missing keys
        """
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        # Filter out comments (keys starting with _)
        overrides = {k: v for k, v in data.items() if not k.startswith('_')}

        return cls(**overrides)

    def to_json(self) -> dict:
        """
        Serialize config for session logging.

        Returns:
            Dict with all config values including derived properties
        """
        return {
            # Core parameters
            'aggressiveness': self.aggressiveness,
            'max_bet_per_trade_usd': self.max_bet_per_trade_usd,
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'max_positions_per_city': self.max_positions_per_city,
            'max_total_positions': self.max_total_positions,

            # Edge classifier
            'edge_confidence_threshold_base': self.edge_confidence_threshold_base,
            'edge_threshold_degf': self.edge_threshold_degf,

            # Order execution
            'maker_timeout_base_seconds': self.maker_timeout_base_seconds,
            'volume_timeout_multiplier': self.volume_timeout_multiplier,
            'volume_lookback_minutes': self.volume_lookback_minutes,

            # Kelly sizing
            'bankroll_usd': self.bankroll_usd,
            'kelly_fraction_base': self.kelly_fraction_base,

            # Cities
            'enabled_cities': self.enabled_cities,

            # Fees
            'maker_fee_pct': self.maker_fee_pct,
            'taker_fee_pct': self.taker_fee_pct,

            # Thresholds
            'min_ev_per_contract_cents': self.min_ev_per_contract_cents,
            'min_bracket_prob': self.min_bracket_prob,

            # Inference
            'inference_cooldown_sec': self.inference_cooldown_sec,

            # Inference strategy
            'inference_mode': self.inference_mode,
            'edge_classifier_cities': self.edge_classifier_cities,
            'fallback_to_threshold': self.fallback_to_threshold,

            # Derived (computed from aggressiveness)
            'effective_confidence_threshold': self.effective_confidence_threshold,
            'effective_kelly_fraction': self.effective_kelly_fraction,
            'effective_maker_timeout_multiplier': self.effective_maker_timeout_multiplier,
        }

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not (0.0 <= self.aggressiveness <= 1.0):
            errors.append(f"aggressiveness must be in [0, 1], got {self.aggressiveness}")

        if self.max_bet_per_trade_usd <= 0:
            errors.append(f"max_bet_per_trade_usd must be positive, got {self.max_bet_per_trade_usd}")

        if self.max_daily_loss_usd <= 0:
            errors.append(f"max_daily_loss_usd must be positive, got {self.max_daily_loss_usd}")

        if self.max_positions_per_city <= 0:
            errors.append(f"max_positions_per_city must be positive, got {self.max_positions_per_city}")

        if self.edge_threshold_degf <= 0:
            errors.append(f"edge_threshold_degf must be positive, got {self.edge_threshold_degf}")

        if len(self.enabled_cities) == 0:
            errors.append("enabled_cities must have at least one city")

        valid_cities = {'chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia'}
        for city in self.enabled_cities:
            if city not in valid_cities:
                errors.append(f"Unknown city '{city}', must be one of {valid_cities}")

        # Validate inference mode
        valid_modes = {'adaptive', 'classifier_only', 'threshold_only'}
        if self.inference_mode not in valid_modes:
            errors.append(f"inference_mode must be one of {valid_modes}, got '{self.inference_mode}'")

        # Validate classifier cities
        for city in self.edge_classifier_cities:
            if city not in valid_cities:
                errors.append(f"Unknown classifier city '{city}', must be one of {valid_cities}")

        return errors

    def __str__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"TradingConfig(\n"
            f"  aggressiveness={self.aggressiveness:.2f}\n"
            f"  confidence_threshold={self.effective_confidence_threshold:.3f}\n"
            f"  kelly_fraction={self.effective_kelly_fraction:.3f}\n"
            f"  maker_timeout_mult={self.effective_maker_timeout_multiplier:.2f}x\n"
            f"  max_bet=${self.max_bet_per_trade_usd:.0f}\n"
            f"  cities={self.enabled_cities}\n"
            f")"
        )
