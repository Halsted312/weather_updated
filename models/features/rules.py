"""
Rule-based meta-features for temperature Δ-models.

This module wraps the deterministic rounding rules from analysis/temperature/rules.py
and computes meta-features that capture:
- Predictions from each rule applied to partial-day data
- Errors vs settlement (for training) or raw predictions (for inference)
- Disagreement signals between rules

These features provide the model with "expert knowledge" about how
different rounding strategies perform, allowing it to learn when
each rule is more or less reliable.

Features computed:
    pred_{rule}_sofar: Prediction from each rule
    err_{rule}_sofar: Error vs settlement (rule_pred - settle_f)
    range_pred_rules_sofar: Range of predictions across all rules
    num_distinct_preds_sofar: Number of unique predictions
    disagree_flag_sofar: 1 if rules disagree, 0 otherwise

Example:
    >>> temps = [90.1, 92.7, 93.4, 93.2]
    >>> fs = compute_rule_features(temps, settle_f=93)
    >>> fs['pred_max_round_sofar']  # round(93.4) = 93
    93
    >>> fs['err_max_round_sofar']   # 93 - 93 = 0
    0
"""

from typing import Optional

from models.features.base import FeatureSet, register_feature_group

# Import the deterministic rules from analysis module
from analysis.temperature.rules import ALL_RULES


@register_feature_group("rules")
def compute_rule_features(
    temps_sofar: list[float],
    settle_f: Optional[int] = None,
) -> FeatureSet:
    """Apply all rules to partial-day temps and compute meta-features.

    Each rule represents a hypothesis about how NWS/ASOS processes
    temperatures into daily highs. By providing rule predictions as
    features, the ML model can learn which rules are reliable in
    different situations.

    Args:
        temps_sofar: List of 5-minute VC temperatures up to snapshot time
        settle_f: Actual settlement (for training). None for inference.

    Returns:
        FeatureSet with rule predictions, errors, and disagreement features
    """
    if not temps_sofar:
        return FeatureSet(name="rules", features={})

    features = {}
    predictions = []

    # Apply each rule
    for rule_name, rule_fn in ALL_RULES.items():
        try:
            pred = rule_fn(temps_sofar)
        except Exception:
            pred = None

        feature_name = f"pred_{rule_name}_sofar"
        features[feature_name] = pred

        if pred is not None:
            predictions.append(pred)

            # Compute error if we have settlement (training mode)
            if settle_f is not None:
                err_name = f"err_{rule_name}_sofar"
                features[err_name] = pred - settle_f
            else:
                features[f"err_{rule_name}_sofar"] = None
        else:
            features[f"err_{rule_name}_sofar"] = None

    # Meta-features about rule agreement
    if predictions:
        features["range_pred_rules_sofar"] = max(predictions) - min(predictions)
        features["num_distinct_preds_sofar"] = len(set(predictions))
        features["disagree_flag_sofar"] = 1 if len(set(predictions)) > 1 else 0
    else:
        features["range_pred_rules_sofar"] = None
        features["num_distinct_preds_sofar"] = None
        features["disagree_flag_sofar"] = None

    return FeatureSet(name="rules", features=features)


def get_consensus_prediction(temps_sofar: list[float]) -> Optional[int]:
    """Get the most common prediction across all rules.

    When rules agree, this gives high-confidence baseline prediction.
    When they disagree, returns the mode (most frequent).

    Args:
        temps_sofar: List of temperatures

    Returns:
        Most common prediction, or None if no predictions
    """
    if not temps_sofar:
        return None

    predictions = []
    for rule_fn in ALL_RULES.values():
        try:
            pred = rule_fn(temps_sofar)
            if pred is not None:
                predictions.append(pred)
        except Exception:
            pass

    if not predictions:
        return None

    # Return mode (most frequent)
    from collections import Counter
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]


def get_rule_agreement_score(temps_sofar: list[float]) -> float:
    """Compute agreement score between rules.

    Returns 1.0 if all rules agree, lower values indicate disagreement.
    Useful as a confidence indicator.

    Args:
        temps_sofar: List of temperatures

    Returns:
        Agreement score in [0, 1] where 1 = full agreement
    """
    if not temps_sofar:
        return 0.0

    predictions = []
    for rule_fn in ALL_RULES.values():
        try:
            pred = rule_fn(temps_sofar)
            if pred is not None:
                predictions.append(pred)
        except Exception:
            pass

    if len(predictions) <= 1:
        return 1.0

    # Agreement = 1 - (range / max possible range)
    # Since rules typically differ by at most 1-2 degrees
    pred_range = max(predictions) - min(predictions)
    max_reasonable_range = 3  # Assume max 3°F disagreement is "reasonable"

    agreement = max(0.0, 1.0 - (pred_range / max_reasonable_range))
    return agreement


def explain_rule_disagreement(temps_sofar: list[float]) -> dict:
    """Explain why rules might disagree.

    Returns diagnostic information useful for understanding
    edge cases where rules give different predictions.

    Args:
        temps_sofar: List of temperatures

    Returns:
        Dict with diagnostic info:
            max_temp, rounded_max, fractional_part, predictions_by_rule
    """
    if not temps_sofar:
        return {}

    max_temp = max(temps_sofar)
    rounded_max = round(max_temp)
    frac_part = max_temp - rounded_max

    predictions = {}
    for rule_name, rule_fn in ALL_RULES.items():
        try:
            predictions[rule_name] = rule_fn(temps_sofar)
        except Exception:
            predictions[rule_name] = None

    return {
        "max_temp": max_temp,
        "rounded_max": rounded_max,
        "fractional_part": frac_part,
        "predictions_by_rule": predictions,
        "num_unique": len(set(p for p in predictions.values() if p is not None)),
    }
