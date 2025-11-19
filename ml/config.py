#!/usr/bin/env python3
"""
Production configuration loader with Pydantic validation.

Loads YAML config files and validates hyperparameters, feature sets,
risk parameters, and training settings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import yaml


class SearchSpace(BaseModel):
    """Hyperparameter search space for Optuna."""
    C_min: float = Field(gt=0, description="Min inverse regularization strength")
    C_max: float = Field(gt=0, description="Max inverse regularization strength")
    l1_ratio_min: float = Field(ge=0, le=1, description="Min L1 ratio for elasticnet")
    l1_ratio_max: float = Field(ge=0, le=1, description="Max L1 ratio for elasticnet")
    class_weight: List[Optional[str]] = Field(
        default=[None, "balanced"],
        description="Class weight options"
    )


class Calibration(BaseModel):
    """Probability calibration settings."""
    method_large: Literal["isotonic", "sigmoid"] = Field(
        default="isotonic",
        description="Calibration method for large calibration sets"
    )
    method_small: Literal["isotonic", "sigmoid"] = Field(
        default="sigmoid",
        description="Calibration method for small calibration sets"
    )
    threshold: int = Field(
        default=1000,
        ge=100,
        description="Calibration set size threshold for method selection"
    )


class RiskParams(BaseModel):
    """Risk management and position sizing parameters."""
    kelly_alpha: float = Field(
        default=0.25,
        gt=0,
        le=1,
        description="Fractional Kelly multiplier"
    )
    max_spread_cents: int = Field(
        default=3,
        ge=0,
        description="Maximum spread in cents (skip if wider)"
    )
    tau_open_cents: float = Field(
        default=1.0,
        ge=0,
        description="Entry edge threshold in cents after costs"
    )
    tau_close_cents: float = Field(
        default=0.5,
        ge=0,
        description="Exit edge threshold in cents"
    )
    slip_per_leg_cents: int = Field(
        default=1,
        ge=0,
        description="Slippage assumption per leg in cents"
    )
    max_bankroll_pct_city_day_side: float = Field(
        default=0.10,
        gt=0,
        le=1,
        description="Max % of capital per city/day/side"
    )


class TrainConfig(BaseModel):
    """
    Production training configuration.

    Validates all model, training, and risk parameters from YAML config files.
    """
    # City and bracket
    city: str = Field(description="City name (e.g., chicago)")
    bracket: Literal["between", "greater", "less"] = Field(
        description="Bracket type"
    )

    # Feature set
    feature_set: Literal["baseline", "ridge_conservative", "elasticnet_rich"] = Field(
        description="Feature set to use"
    )

    # Training window parameters
    train_days: int = Field(default=90, ge=1, description="Training window size in days")
    test_days: int = Field(default=7, ge=1, description="Test window size in days")
    step_days: int = Field(default=7, ge=1, description="Step size for walk-forward windows")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")

    # Hyperparameter tuning
    penalties: List[Literal["l1", "l2", "elasticnet"]] = Field(
        default=["elasticnet"],
        description="Penalty types to search"
    )
    trials: int = Field(default=40, ge=1, description="Optuna trials per window")
    cv_splits: int = Field(default=4, ge=2, description="GroupKFold CV splits")

    # Search space
    search_space: SearchSpace = Field(description="Hyperparameter search ranges")

    # Calibration
    calibration: Calibration = Field(description="Calibration settings")

    # Blending
    blend_weight: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Model weight for opinion pooling"
    )

    # Risk management
    risk: RiskParams = Field(description="Risk and position sizing parameters")

    # VC feature settings
    use_vc_minutes: bool = Field(
        default=True,
        description="Whether this city uses VC minute features"
    )
    excluded_vc_cities: List[str] = Field(
        default_factory=list,
        description="Cities excluded from VC features"
    )

    # Reproducibility
    seed: int = Field(default=42, description="Random seed")

    # Pilot provenance (optional)
    pilot_dir: Optional[str] = Field(
        default=None,
        description="Path to pilot model directory"
    )
    pilot_windows: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of pilot windows"
    )
    pilot_total_test_rows: Optional[int] = Field(
        default=None,
        ge=1,
        description="Total test rows in pilot"
    )

    @field_validator("penalties")
    @classmethod
    def validate_penalties(cls, v):
        """Ensure at least one penalty is specified."""
        if not v:
            raise ValueError("At least one penalty must be specified")
        return v

    @field_validator("blend_weight")
    @classmethod
    def validate_blend_weight(cls, v):
        """Validate blend weight is in valid range."""
        if not (0 <= v <= 1):
            raise ValueError("blend_weight must be between 0 and 1")
        return v


def load_config(config_path: str) -> TrainConfig:
    """
    Load and validate training config from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated TrainConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate with Pydantic
    config = TrainConfig(**config_dict)

    return config


def main():
    """Demo: Load and validate ElasticNet Chicago/between config."""
    print("\n" + "="*60)
    print("Config Loader Demo")
    print("="*60 + "\n")

    # Load production config
    config_path = "configs/elasticnet_chi_between.yaml"

    try:
        config = load_config(config_path)

        print(f"Loaded config: {config_path}")
        print(f"\nCity/Bracket: {config.city} / {config.bracket}")
        print(f"Feature set: {config.feature_set}")
        print(f"Training window: {config.train_days}→{config.test_days} days")
        print(f"Penalties: {config.penalties}")
        print(f"Trials: {config.trials}")
        print(f"Calibration: {config.calibration.method_large} (N≥{config.calibration.threshold}), "
              f"{config.calibration.method_small} (N<{config.calibration.threshold})")
        print(f"Blend weight: {config.blend_weight} (model), "
              f"{1 - config.blend_weight:.2f} (market)")
        print(f"Kelly alpha: {config.risk.kelly_alpha}")
        print(f"VC features: {config.use_vc_minutes}")
        print(f"Excluded cities: {config.excluded_vc_cities}")

        if config.pilot_dir:
            print(f"\nPilot provenance:")
            print(f"  Directory: {config.pilot_dir}")
            print(f"  Windows: {config.pilot_windows}")
            print(f"  Test rows: {config.pilot_total_test_rows}")

        print("\n✓ Config validation passed")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
