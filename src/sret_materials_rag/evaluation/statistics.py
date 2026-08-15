from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationResult:
    value: float | None
    n: int
    ci_low: float | None = None
    ci_high: float | None = None


def pearson_r(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = average_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson_r(_rank(x), _rank(y))


def bootstrap_ci(
    x: list[float],
    y: list[float],
    *,
    statistic=pearson_r,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    if len(x) < 2:
        return None, None

    rng = np.random.default_rng(seed)
    values: list[float] = []
    x_arr = np.array(x)
    y_arr = np.array(y)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x_arr), len(x_arr))
        value = statistic(x_arr[indices].tolist(), y_arr[indices].tolist())
        if value is not None and not math.isnan(value):
            values.append(value)

    if not values:
        return None, None
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def correlation_with_ci(
    x: list[float],
    y: list[float],
    *,
    statistic=pearson_r,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> CorrelationResult:
    value = statistic(x, y)
    ci_low, ci_high = bootstrap_ci(
        x,
        y,
        statistic=statistic,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return CorrelationResult(value=value, n=len(x), ci_low=ci_low, ci_high=ci_high)

