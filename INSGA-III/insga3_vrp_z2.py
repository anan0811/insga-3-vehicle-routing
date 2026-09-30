from typing import Iterable, List, Sequence

try:
    from .insga3_vrp_load_flow import build_customer_index, split_sequence_by_capacity
except ImportError:
    from insga3_vrp_load_flow import build_customer_index, split_sequence_by_capacity


def decode_routes_for_z2(
    sequence: Sequence[object],
    instance: Iterable[list],
    vehicle_capacity: float,
) -> List[List[object]]:
    customer_index = build_customer_index(instance)
    return split_sequence_by_capacity(list(sequence), customer_index, vehicle_capacity)


def count_routes_as_vehicles(routes: Sequence[Sequence[object]]) -> int:
    return sum(1 for route in routes if route)


def evaluate_z2_from_sequence(
    sequence: Sequence[object],
    instance: Iterable[list],
    vehicle_capacity: float,
):
    routes = decode_routes_for_z2(
        sequence=sequence,
        instance=instance,
        vehicle_capacity=vehicle_capacity,
    )
    return routes, count_routes_as_vehicles(routes)
