from typing import Dict, Iterable, List, Tuple

EPS = 1e-9


def _to_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_customer_index(instance: Iterable[list]) -> Dict[object, list]:
    return {row[0]: row for row in instance if row}


def get_delivery_pickup(row: list) -> Tuple[float, float]:
    delivery = _to_float(row[5]) if len(row) > 5 else 0.0
    pickup = _to_float(row[6]) if len(row) > 6 else 0.0
    return delivery, pickup


def get_aggregated_demand(row: list) -> float:
    delivery, pickup = get_delivery_pickup(row)
    return delivery + pickup


def route_aggregated_demand(
    route: List[object], customer_index: Dict[object, list]
) -> float:
    total = 0.0
    for customer_id in route:
        row = customer_index.get(customer_id)
        if row is None:
            continue
        total += get_aggregated_demand(row)
    return total


def route_load_profile(
    route: List[object], customer_index: Dict[object, list], capacity: float
) -> dict:
    capacity = float(capacity)
    start_load = 0.0
    for customer_id in route:
        row = customer_index.get(customer_id)
        if row is None:
            continue
        delivery, _pickup = get_delivery_pickup(row)
        start_load += delivery

    current_load = start_load
    loads = [current_load]
    max_over = max(0.0, current_load - capacity)
    max_under = max(0.0, -current_load)

    for customer_id in route:
        row = customer_index.get(customer_id)
        if row is None:
            continue
        delivery, pickup = get_delivery_pickup(row)
        current_load = current_load - delivery + pickup
        loads.append(current_load)
        max_over = max(max_over, current_load - capacity)
        max_under = max(max_under, -current_load)

    feasible = (max_over <= EPS) and (max_under <= EPS)
    return {
        "feasible": feasible,
        "start_load": start_load,
        "end_load": current_load,
        "loads": loads,
        "max_over": max_over,
        "max_under": max_under,
    }


def is_route_capacity_feasible(
    route: List[object], customer_index: Dict[object, list], capacity: float
) -> bool:
    if not route:
        return True
    profile = route_load_profile(route, customer_index, capacity)
    return bool(profile["feasible"])


def split_sequence_by_capacity(
    sequence: List[object], customer_index: Dict[object, list], capacity: float
) -> List[List[object]]:
    routes: List[List[object]] = []
    current: List[object] = []

    for customer_id in sequence:
        if not current:
            current = [customer_id]
            continue

        candidate_route = current + [customer_id]
        if is_route_capacity_feasible(candidate_route, customer_index, capacity):
            current = candidate_route
            continue

        routes.append(current)
        current = [customer_id]

    if current:
        routes.append(current)
    return routes
