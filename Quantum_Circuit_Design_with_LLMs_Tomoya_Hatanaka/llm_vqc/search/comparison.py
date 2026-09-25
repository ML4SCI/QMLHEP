"""Shared candidate-comparison rule every search arm must use.

A single function, used identically by every arm's "is this new candidate
better than my current best" and "which of these do I keep" decisions, so
that no arm can silently apply a different tie-breaking rule than another
(that would be an unmatchable, arm-private piece of the search-space
parity the master plan requires -- Section 6.4).

**Tie-breaking is deterministic and strict:** a tie (equal metric value)
never replaces the incumbent. Combined with each arm evaluating proposals
in a fixed, seeded order, this means "first candidate to reach a given
metric value wins ties" -- fully reproducible given the run seed.
"""

from __future__ import annotations


def is_strictly_better(
    candidate_metric: float, current_best_metric: float | None, lower_is_better: bool
) -> bool:
    """Whether `candidate_metric` should replace `current_best_metric`.

    `current_best_metric=None` means "no candidate has been accepted yet"
    -- always true in that case. Ties (`candidate_metric == current_best_metric`)
    are never an improvement, which is what makes tie-breaking deterministic.
    """
    if current_best_metric is None:
        return True
    if lower_is_better:
        return candidate_metric < current_best_metric
    return candidate_metric > current_best_metric
