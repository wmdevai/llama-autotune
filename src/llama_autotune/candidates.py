"""Candidate-generation helpers for the optimizer.

Pure functions that produce candidate parameter values for the local search
(Stage B) and Bayesian refinement (Stage C). They are kept separate from the
``Optimizer`` class so they can be unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any

import optuna

from .search_space import ParamDef


def strategic_ctx_values(
    param: ParamDef,
    current_value: Any,
    count: int,
    include_current: bool = True,
) -> list[Any]:
    """Generate strategic context-size candidates around the current value.

    When ``include_current`` is true, the current context is preserved as
    the first candidate. Otherwise, the requested budget is spent only on
    alternative context values. Candidates above the current value are
    preferred and distributed progressively towards the maximum supported
    context. At the upper bound, refinement proceeds downward in aligned
    steps.
    """
    if count <= 0:
        return []

    if current_value is None:
        current_value = param.low

    current = max(
        param.low,
        min(param.high, current_value),
    )

    step = param.step or 1

    current_value_normalized = (
        int(current)
        if isinstance(param.low, int)
        else current
    )

    values: list[Any] = []

    if include_current:
        values.append(current_value_normalized)

    if current >= param.high:
        distance = 1

        while len(values) < count:
            candidate = current - distance * step

            if candidate < param.low:
                break

            value = (
                int(candidate)
                if isinstance(param.low, int)
                else candidate
            )

            if value not in values:
                values.append(value)

            distance += 1

        return values[:count]

    remaining = count - len(values)

    if remaining <= 0:
        return values[:count]

    # When the current value is excluded, Stage B should spend the
    # budget on real alternatives starting from the nearest aligned
    # value. Preserve the maximum context as the final candidate when
    # more than one alternative is requested.
    if not include_current:
        candidate = current + step

        while (
            len(values) < count - 1
            and candidate < param.high
        ):
            value = (
                int(candidate)
                if isinstance(param.low, int)
                else candidate
            )

            if value not in values:
                values.append(value)

            candidate += step

        if (
            len(values) < count
            and param.high not in values
        ):
            values.append(
                int(param.high)
                if isinstance(param.low, int)
                else param.high
            )

        return values[:count]

    distance = param.high - current

    for index in range(1, remaining + 1):
        target = current + (
            distance * index / remaining
        )

        offset = target - param.low

        aligned = (
            param.low
            + ((offset + step - 1) // step) * step
        )

        if aligned > param.high:
            aligned = param.high

        value = (
            int(aligned)
            if isinstance(param.low, int)
            else aligned
        )

        if (
            value != current_value_normalized
            and value not in values
        ):
            values.append(value)

    candidate = current + step

    while (
        len(values) < count
        and candidate <= param.high
    ):
        value = (
            int(candidate)
            if isinstance(param.low, int)
            else candidate
        )

        if (
            value != current_value_normalized
            and value not in values
        ):
            values.append(value)

        candidate += step

    return values[:count]


def local_values(
    param: ParamDef,
    current_value: Any,
    count: int,
) -> list[Any]:
    """Generate values locally around the current parameter value.

    Numeric values are explored in alternating order around the current
    value: one step below, one step above, then progressively farther
    away. Values are kept within the parameter bounds and duplicates are
    avoided. If the current value lies outside the search-space bounds,
    it is clamped only as the center for local candidate generation.
    """
    if count <= 0:
        return []

    if param.is_categorical and param.categories:
        if current_value in param.categories:
            start = param.categories.index(current_value)
            values = []
            distance = 1

            while (
                len(values) < count
                and (
                    start - distance >= 0
                    or start + distance < len(param.categories)
                )
            ):
                for index in (
                    start - distance,
                    start + distance,
                ):
                    if (
                        0 <= index < len(param.categories)
                        and len(values) < count
                    ):
                        values.append(param.categories[index])
                distance += 1

            return values

        return param.categories[:count]

    if current_value is None:
        current_value = param.low

    above_high = current_value > param.high

    current = max(
        param.low,
        min(param.high, current_value),
    )

    # A value above the search-space maximum can represent a semantic
    # "maximum" setting, such as n_gpu_layers=999 for full offload.
    # For n_gpu_layers, treat both that representation and the explicit
    # numeric maximum as refinement around the full-offload boundary.
    if (
        param.name == "n_gpu_layers"
        and current >= param.high
    ):
        step = 1
    else:
        step = param.step or max(
            1,
            (param.high - param.low) // max(count, 1),
        )

    values = []

    if above_high and count > 0:
        values.append(
            int(current)
            if isinstance(param.low, int)
            else current
        )

    distance = 1

    while len(values) < count:
        added = False

        for candidate in (
            current - distance * step,
            current + distance * step,
        ):
            if (
                param.low <= candidate <= param.high
                and candidate not in values
                and candidate != current
                and len(values) < count
            ):
                values.append(
                    int(candidate)
                    if isinstance(param.low, int)
                    else candidate
                )
                added = True

        if (
            current - distance * step < param.low
            and current + distance * step > param.high
        ):
            break

        if not added and (
            current - distance * step < param.low
            and current + distance * step > param.high
        ):
            break

        distance += 1

    return values


def stage_b_ngl_values(
    param: ParamDef,
    current_value: Any,
    count: int,
    include_current: bool = True,
) -> list[Any]:
    """Generate Stage B candidates for GPU layer offload.

    Optionally preserves a semantic full-offload value from the current
    configuration when it exceeds the numeric model layer range, then
    fills the remaining allocation with the highest concrete layer counts.
    """
    if count <= 0:
        return []

    values: list[Any] = []

    if (
        include_current
        and current_value is not None
        and current_value > param.high
    ):
        values.append(current_value)

    candidate = int(param.high)

    while len(values) < count and candidate >= int(param.low):
        if candidate not in values:
            values.append(candidate)
        candidate -= 1

    return values


def grid_values(param: ParamDef, count: int) -> list[Any]:
    """Generate up to ``count`` evenly-spaced values for a parameter.

    For categorical parameters the first ``count`` categories are
    returned.  For numeric parameters with a step size the values are
    produced by repeated addition; otherwise a uniform step is computed.
    """
    if param.is_categorical and param.categories:
        return param.categories[:count]
    if param.step:
        values = []
        v = param.low
        while v <= param.high and len(values) < count:
            values.append(int(v) if isinstance(param.low, int) else v)
            v += param.step
        return values
    low = int(param.low)
    high = int(param.high)
    step = max(1, (high - low) // (count - 1))
    return list(range(low, high + 1, step))


def sample_param(
    trial: optuna.Trial,
    name: str,
    pdef: ParamDef,
) -> Any:
    """Sample a parameter value for an Optuna trial.

    Delegates to the appropriate ``suggest_*`` method based on whether the
    parameter is categorical or integer-valued.
    """
    if pdef.is_categorical and pdef.categories:
        return trial.suggest_categorical(name, pdef.categories)
    if pdef.step:
        return trial.suggest_int(
            name, int(pdef.low), int(pdef.high), step=int(pdef.step)
        )
    return trial.suggest_int(name, int(pdef.low), int(pdef.high))
