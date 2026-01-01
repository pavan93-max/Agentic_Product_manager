from typing import Dict, Literal
import logging

logger = logging.getLogger(__name__)
SHIP_THRESHOLD = 0.95
ROLLBACK_THRESHOLD = 0.60


def decide(
    result: Dict[str, float],
    ship_threshold: float = SHIP_THRESHOLD,
    rollback_threshold: float = ROLLBACK_THRESHOLD
) -> Literal["SHIP", "ROLLBACK", "ITERATE"]:
    if ship_threshold <= rollback_threshold:
        raise ValueError("ship_threshold must be greater than rollback_threshold")
    if "prob_treatment_better" not in result:
        raise KeyError("result must contain 'prob_treatment_better' key")
    p = result["prob_treatment_better"]
    if p >= ship_threshold:
        decision = "SHIP"
    elif p <= rollback_threshold:
        decision = "ROLLBACK"
    else:
        decision = "ITERATE"
    logger.info(f"Decision: {decision} (P={p:.4f}, threshold=[{rollback_threshold}, {ship_threshold}])")
    return decision
