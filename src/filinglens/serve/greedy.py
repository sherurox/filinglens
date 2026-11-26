"""Greedy baseline router for FL-M5 T10 (hand-written; compared against the LP in serve/router.py).

Rule (bench/m5_router_rule.json, greedy_baseline): classes in the order `traffic` lists them; for
each, the eligible backends whose own accuracy meets the floor, cheapest first; give each as much
of the class's remaining traffic as its remaining capacity allows, then the next; whatever is left
is unrouted. Capacity is shared across classes and consumed in class order. Greedy never blends a
below-floor backend in (the LP may: it only needs the share-weighted accuracy to meet the floor).
"""

from __future__ import annotations


def greedy_route(
    traffic: dict[str, float],
    options: dict[str, list[dict]],
    floors: dict[str, float | None],
    capacity: dict[str, float],
    usage: dict[tuple[str, str], float],
) -> dict:
    """Route each class's daily traffic greedily.

    traffic:  class -> requests per day, in routing order
    options:  class -> eligible backends, each {"backend", "accuracy", "usd_per_request"}
    floors:   class -> minimum accuracy, or None for no floor
    capacity: backend -> capacity units per day (Gemini: requests; local: busy seconds)
    usage:    (class, backend) -> capacity units one request uses

    Returns {"assigned": {(class, backend): requests}, "unrouted": {class: requests},
             "cost_usd": float}.
    """
    # 1: private copy of capacity, so the caller's dict is never modified.
    remaining = dict(capacity)

    # 2: the three outputs.
    assigned: dict[tuple[str, str], float] = {}
    unrouted: dict[str, float] = {}
    cost = 0.0

    # 3: classes in the order `traffic` lists them.
    for cls, demand in traffic.items():
        left = demand  # 3a

        # 3b: drop backends whose own accuracy is below the floor (floor None keeps all).
        # floors[cls], not .get(): a class missing from `floors` is a caller error, and a
        # silent "no floor" would route it to below-floor backends.
        floor = floors[cls]
        eligible = [
            opt for opt in options.get(cls, []) if floor is None or opt["accuracy"] >= floor
        ]

        # 3c: cheapest first (sorted() is stable, so ties keep their listed order).
        eligible = sorted(eligible, key=lambda opt: opt["usd_per_request"])

        # 3d: fill each backend up to its remaining capacity, then spill to the next.
        for opt in eligible:
            if left <= 0:
                break
            backend = opt["backend"]
            per_request = usage[(cls, backend)]
            room = remaining.get(backend, 0.0)
            # A request that uses no capacity is never capacity-limited.
            fits = room / per_request if per_request > 0 else left
            take = min(fits, left)
            if take > 0:
                # add, don't overwrite: a backend listed twice keeps both amounts
                assigned[(cls, backend)] = assigned.get((cls, backend), 0.0) + take
                # max() guards against a -1e-16 left over from float subtraction.
                remaining[backend] = max(0.0, room - take * per_request)
                left -= take
                cost += take * opt["usd_per_request"]

        # 3e: whatever no backend could take.
        unrouted[cls] = left

    # 4
    return {"assigned": assigned, "unrouted": unrouted, "cost_usd": cost}
