from typing import Callable, Dict, Iterable, List, Sequence

try:
    from .insga3_vrp_load_flow import build_customer_index
    from .insga3_vrp_z2 import count_routes_as_vehicles
except ImportError:
    from insga3_vrp_load_flow import build_customer_index
    from insga3_vrp_z2 import count_routes_as_vehicles


INFEASIBLE_OBJECTIVE = 1e12
SATELLITE_WINDOW_SPAN_MINUTES = 1440.0
ENFORCE_SATELLITE_TIME_WINDOW = False


def _safe_round(value: float, ndigits: int) -> float:
    return round(float(value), ndigits)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _customer_latest_time(row):
    if row is None or len(row) <= 4:
        return None
    return _safe_float(row[4])


def evaluate_single_period_fitness(
    routes: Sequence[Sequence[object]],
    se_vehicle_speed: float,
    instance: Iterable[list],
    satellite,
    time_start: float,
    distance_fn: Callable,
    time_cost_fn: Callable,
    carbon_contribution_fn: Callable,
    se_vehicle_capacity: float = None,
):
    customer_index: Dict[object, list] = build_customer_index(instance)
    total_cost = 0.0
    total_carbon = 0.0
    num_vehicle = count_routes_as_vehicles(routes)
    hard_violation_count = 0
    satellite_latest = None
    if ENFORCE_SATELLITE_TIME_WINDOW:
        # Hard feasibility check for second-echelon satellite time window.
        # The practical upper bound is inferred from the latest customer upper-time-window
        # among customers served in this period.
        for route in routes:
            for customer_id in route:
                row = customer_index.get(customer_id)
                latest = _customer_latest_time(row)
                if latest is None:
                    continue
                if satellite_latest is None or latest > satellite_latest:
                    satellite_latest = latest
        lower_bound_latest = float(time_start) + SATELLITE_WINDOW_SPAN_MINUTES
        if satellite_latest is None:
            satellite_latest = lower_bound_latest
        else:
            satellite_latest = max(float(satellite_latest), lower_bound_latest)

    for route in routes:
        if not route:
            continue

        route_cost = 0.0
        route_carbon = 0.0
        time_current = time_start

        first_customer = customer_index.get(route[0])
        if first_customer is None:
            continue

        leg_distance = distance_fn(
            satellite[1], satellite[0], first_customer[2], first_customer[1]
        )
        time_current, route_cost = time_cost_fn(
            time_current, first_customer, leg_distance, se_vehicle_speed, route_cost
        )
        route_carbon += carbon_contribution_fn(leg_distance, se_vehicle_speed)

        for prev_id, curr_id in zip(route, route[1:]):
            prev_customer = customer_index.get(prev_id)
            curr_customer = customer_index.get(curr_id)
            if prev_customer is None or curr_customer is None:
                continue
            leg_distance = distance_fn(
                curr_customer[2], curr_customer[1], prev_customer[2], prev_customer[1]
            )
            time_current, route_cost = time_cost_fn(
                time_current, curr_customer, leg_distance, se_vehicle_speed, route_cost
            )
            route_carbon += carbon_contribution_fn(leg_distance, se_vehicle_speed)

        last_customer = customer_index.get(route[-1])
        if last_customer is not None:
            leg_distance = distance_fn(
                last_customer[2], last_customer[1], satellite[1], satellite[0]
            )
            route_cost += leg_distance
            route_carbon += carbon_contribution_fn(leg_distance, se_vehicle_speed)
            time_current += (leg_distance / max(float(se_vehicle_speed), 1e-6)) * 60.0

        total_cost += route_cost
        total_carbon += route_carbon

        if (
            ENFORCE_SATELLITE_TIME_WINDOW
            and satellite_latest is not None
            and time_current > satellite_latest + 1e-9
        ):
            hard_violation_count += 1

    if hard_violation_count > 0:
        # No additional penalty term: infeasible candidates are directly discarded.
        return [INFEASIBLE_OBJECTIVE, INFEASIBLE_OBJECTIVE, INFEASIBLE_OBJECTIVE]

    # Objective order swapped: z1=total_cost, z2=num_vehicle, z3=carbon
    return [_safe_round(total_cost, 0), num_vehicle, _safe_round(total_carbon, 6)]


def evaluate_period_pair_fitness(
    morning_routes: Sequence[Sequence[object]],
    afternoon_routes: Sequence[Sequence[object]],
    se_vehicle_speed: float,
    instance: Iterable[list],
    satellite_coordinates,
    distance_fn: Callable,
    time_cost_fn: Callable,
    carbon_contribution_fn: Callable,
    se_vehicle_capacity: float = None,
):
    morning = evaluate_single_period_fitness(
        routes=morning_routes,
        se_vehicle_speed=se_vehicle_speed,
        instance=instance,
        satellite=satellite_coordinates,
        time_start=540,
        distance_fn=distance_fn,
        time_cost_fn=time_cost_fn,
        carbon_contribution_fn=carbon_contribution_fn,
        se_vehicle_capacity=se_vehicle_capacity,
    )
    afternoon = evaluate_single_period_fitness(
        routes=afternoon_routes,
        se_vehicle_speed=se_vehicle_speed,
        instance=instance,
        satellite=satellite_coordinates,
        time_start=720,
        distance_fn=distance_fn,
        time_cost_fn=time_cost_fn,
        carbon_contribution_fn=carbon_contribution_fn,
        se_vehicle_capacity=se_vehicle_capacity,
    )
    return morning, afternoon
