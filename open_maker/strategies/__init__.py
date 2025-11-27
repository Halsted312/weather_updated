"""
Strategy registry for open_maker.

Each strategy is registered with:
- strategy_id: unique string identifier
- strategy_class: class implementing StrategyBase
- params_class: dataclass for strategy parameters
"""

from typing import Dict, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StrategyBase, StrategyParamsBase

# Registry populated by strategy modules on import
STRATEGY_REGISTRY: Dict[str, Tuple[Type["StrategyBase"], Type["StrategyParamsBase"]]] = {}


def register_strategy(
    strategy_id: str,
    strategy_class: Type["StrategyBase"],
    params_class: Type["StrategyParamsBase"],
) -> None:
    """Register a strategy in the global registry."""
    STRATEGY_REGISTRY[strategy_id] = (strategy_class, params_class)


def get_strategy(strategy_id: str) -> Tuple[Type["StrategyBase"], Type["StrategyParamsBase"]]:
    """
    Get strategy class and params class by ID.

    Args:
        strategy_id: Strategy identifier (e.g., "open_maker_base")

    Returns:
        Tuple of (strategy_class, params_class)

    Raises:
        KeyError: If strategy_id not found
    """
    if strategy_id not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy: {strategy_id}. Available: {available}")
    return STRATEGY_REGISTRY[strategy_id]


def list_strategies() -> list:
    """Return list of registered strategy IDs."""
    return sorted(STRATEGY_REGISTRY.keys())


# Import strategies to trigger registration
from .base import BaseStrategy, OpenMakerParams  # noqa: E402
from .next_over import NextOverStrategy, NextOverParams  # noqa: E402
from .curve_gap import CurveGapStrategyV2, CurveGapParams, CurveGapDecision  # noqa: E402
