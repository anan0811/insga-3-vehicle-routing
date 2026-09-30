from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _reorder_satellite_candidates(
    candidates: Sequence[int], preferred: Optional[int]
) -> List[int]:
    uniq = sorted({int(x) for x in candidates})
    if preferred is None or preferred not in uniq:
        return uniq
    return sorted(
        uniq, key=lambda x: (0 if x == preferred else 1, abs(x - preferred), x)
    )


@dataclass(frozen=True)
class DayInput:
    day: int
    customer_start: int
    customer_end: int
    crossover_probability: float
    mutation_probability: float
    satellite_candidates: List[int]
    se_vehicle_capacity: float
    daily_depot_inbound: float
    daily_satellite_inbound: float
    raw_day_demand: float
    effective_day_demand: float


@dataclass(frozen=True)
class DayPlan:
    day: int
    mode: str
    customer_start: int
    customer_end: int
    input_crossover_probability: float
    input_mutation_probability: float
    effective_crossover_probability: float
    effective_mutation_probability: float
    satellite_candidates: List[int]
    prev_best_satellite: Optional[int]
    carry_pressure: float


@dataclass(frozen=True)
class DayState:
    day: int
    best_satellite: int
    best_total_cost: float
    best_carbon_objective: float
    best_adjusted_total_cost: float
    best_adjusted_carbon_objective: float
    best_vehicle_count: float
    best_satellite_demand_total: float
    carry_pressure: float
    cross_day_penalty_total: float
    hard_violation_count: int
    start_depot_inventory: float
    start_satellite_inventory: float
    start_backlog: float
    input_depot_inbound: float
    input_satellite_inbound: float
    transfer_depot_to_satellite: float
    raw_day_demand: float
    effective_day_demand: float
    served_today: float
    unserved_today: float
    end_depot_inventory: float
    end_satellite_inventory: float
    end_backlog: float
    inventory_negative_flag: int
    unserved_exists_flag: int


def build_day_input(
    day: int,
    customer_start: int,
    customer_end: int,
    crossover_probability: float,
    mutation_probability: float,
    satellite_candidates: Sequence[int],
    se_vehicle_capacity: float,
    daily_depot_inbound: Optional[float] = None,
    daily_satellite_inbound: Optional[float] = None,
    raw_day_demand: Optional[float] = None,
    effective_day_demand: Optional[float] = None,
) -> DayInput:
    cap = float(se_vehicle_capacity)
    depot_in = (
        float(daily_depot_inbound) if daily_depot_inbound is not None else cap * 8.0
    )
    sat_in = (
        float(daily_satellite_inbound)
        if daily_satellite_inbound is not None
        else cap * 2.0
    )
    raw_demand = max(0.0, float(raw_day_demand if raw_day_demand is not None else 0.0))
    effective_demand = (
        float(effective_day_demand) if effective_day_demand is not None else raw_demand
    )
    effective_demand = max(raw_demand, max(0.0, effective_demand))
    return DayInput(
        day=int(day),
        customer_start=int(customer_start),
        customer_end=int(customer_end),
        crossover_probability=float(crossover_probability),
        mutation_probability=float(mutation_probability),
        satellite_candidates=[int(x) for x in satellite_candidates],
        se_vehicle_capacity=cap,
        daily_depot_inbound=max(0.0, depot_in),
        daily_satellite_inbound=max(0.0, sat_in),
        raw_day_demand=raw_demand,
        effective_day_demand=effective_demand,
    )


def merge_day_input_with_previous_state(
    day_input: DayInput,
    previous_state: Optional[DayState],
    fast_run: bool = False,
) -> DayPlan:
    sat_candidates = (
        [1] if fast_run else sorted({int(x) for x in day_input.satellite_candidates})
    )
    if not sat_candidates:
        sat_candidates = [1]
    if previous_state is None:
        return DayPlan(
            day=day_input.day,
            mode="A",
            customer_start=day_input.customer_start,
            customer_end=day_input.customer_end,
            input_crossover_probability=day_input.crossover_probability,
            input_mutation_probability=day_input.mutation_probability,
            effective_crossover_probability=day_input.crossover_probability,
            effective_mutation_probability=day_input.mutation_probability,
            satellite_candidates=sat_candidates,
            prev_best_satellite=None,
            carry_pressure=0.0,
        )

    pressure = _clamp(previous_state.carry_pressure, 0.0, 2.0)
    # NSGA2-aligned behavior: daily cp/mp use day input values directly.
    effective_cp = day_input.crossover_probability
    effective_mp = day_input.mutation_probability
    return DayPlan(
        day=day_input.day,
        mode="B",
        customer_start=day_input.customer_start,
        customer_end=day_input.customer_end,
        input_crossover_probability=day_input.crossover_probability,
        input_mutation_probability=day_input.mutation_probability,
        effective_crossover_probability=effective_cp,
        effective_mutation_probability=effective_mp,
        satellite_candidates=sat_candidates,
        prev_best_satellite=previous_state.best_satellite,
        carry_pressure=pressure,
    )


def repair_day_plan_for_consistency(
    day_plan: DayPlan,
    previous_state: Optional[DayState],
    fast_run: bool = False,
) -> DayPlan:
    cp = _clamp(day_plan.input_crossover_probability, 0.0, 1.0)
    mp = _clamp(day_plan.input_mutation_probability, 0.0, 1.0)
    sat_candidates = sorted({int(x) for x in day_plan.satellite_candidates})
    if fast_run:
        sat_candidates = [1]
    if not sat_candidates:
        sat_candidates = [1]

    return replace(
        day_plan,
        effective_crossover_probability=cp,
        effective_mutation_probability=mp,
        satellite_candidates=sat_candidates,
    )


def evaluate_cross_day_constraints(
    day_plan: DayPlan,
    previous_state: Optional[DayState],
    candidate_result: Dict[str, float],
) -> Dict[str, object]:
    reasons: List[str] = []
    hard_violations = 0
    penalty = 0.0

    demand_total = float(candidate_result.get("satellite_demand_total", 0.0))
    vehicles = max(1.0, float(candidate_result.get("vehicles", 0.0)))
    se_capacity = max(1.0, float(candidate_result.get("se_vehicle_capacity", 0.0)))
    service_capacity = max(1.0, vehicles * se_capacity)
    overload = max(0.0, demand_total - service_capacity)
    if overload > 0.0:
        overload_ratio = overload / service_capacity
        hard_violations += 1
        p = 1_000_000.0 * (1.0 + overload_ratio)
        penalty += p
        reasons.append(f"capacity_overload={overload:.3f}")

    carbon = float(candidate_result.get("carbon", 0.0))
    if carbon >= 1e8:
        hard_violations += 1
        p = 500_000.0
        penalty += p
        reasons.append("invalid_carbon_domain")

    if previous_state is not None and day_plan.mode == "B":
        sat_now = int(candidate_result.get("satellite", 1))
        sat_jump = abs(sat_now - int(previous_state.best_satellite))
        if sat_jump > 1:
            p = 100_000.0 * sat_jump
            penalty += p
            reasons.append(f"satellite_jump={sat_jump}")

        vehicle_now = float(candidate_result.get("vehicles", 0.0))
        vehicle_jump = abs(vehicle_now - float(previous_state.best_vehicle_count))
        if vehicle_jump > 2.0:
            p = 50_000.0 * (vehicle_jump - 2.0)
            penalty += p
            reasons.append(f"vehicle_jump={vehicle_jump:.3f}")

    adjusted_total_cost = float(candidate_result.get("total_cost", 0.0)) + penalty
    adjusted_carbon = carbon + penalty

    return {
        "adjusted_total_cost": adjusted_total_cost,
        "adjusted_carbon": adjusted_carbon,
        "penalty_total": penalty,
        "hard_violations": hard_violations,
        "reasons": reasons,
    }


def _calc_state_transition(
    day_input: DayInput,
    previous_state: Optional[DayState],
    best_result: Dict[str, float],
) -> Dict[str, float]:
    if previous_state is None:
        start_depot = max(0.0, day_input.daily_depot_inbound)
        start_sat = max(0.0, day_input.daily_satellite_inbound)
        start_backlog = 0.0
    else:
        start_depot = max(
            0.0, previous_state.end_depot_inventory + day_input.daily_depot_inbound
        )
        start_sat = max(
            0.0,
            previous_state.end_satellite_inventory + day_input.daily_satellite_inbound,
        )
        start_backlog = max(0.0, previous_state.end_backlog)

    raw_demand = max(0.0, float(day_input.raw_day_demand))
    total_required = max(raw_demand, float(day_input.effective_day_demand))
    transfer_needed = max(0.0, total_required - start_sat)
    transfer = min(start_depot, transfer_needed)

    available_sat = start_sat + transfer
    served = min(available_sat, total_required)
    unserved = max(0.0, total_required - served)

    end_sat = max(0.0, available_sat - served)
    end_depot = max(0.0, start_depot - transfer)
    end_backlog = unserved

    inventory_negative_flag = int((end_sat < -1e-9) or (end_depot < -1e-9))
    unserved_exists_flag = int(end_backlog > 1e-9)

    return {
        "start_depot_inventory": start_depot,
        "start_satellite_inventory": start_sat,
        "start_backlog": start_backlog,
        "input_depot_inbound": float(day_input.daily_depot_inbound),
        "input_satellite_inbound": float(day_input.daily_satellite_inbound),
        "transfer_depot_to_satellite": transfer,
        "raw_day_demand": raw_demand,
        "effective_day_demand": total_required,
        "served_today": served,
        "unserved_today": unserved,
        "end_depot_inventory": end_depot,
        "end_satellite_inventory": end_sat,
        "end_backlog": end_backlog,
        "inventory_negative_flag": inventory_negative_flag,
        "unserved_exists_flag": unserved_exists_flag,
    }


def build_next_day_state(
    day_input: DayInput,
    day_plan: DayPlan,
    previous_state: Optional[DayState],
    best_result: Dict[str, float],
) -> DayState:
    best_sat = int(best_result["satellite"])
    demand_total = float(best_result.get("satellite_demand_total", 0.0))
    se_capacity = float(best_result.get("se_vehicle_capacity", 0.0))
    vehicles = max(1.0, float(best_result.get("vehicles", 0.0)))
    service_capacity = max(1.0, vehicles * max(1.0, se_capacity))

    state = _calc_state_transition(day_input, previous_state, best_result)
    overload_ratio = max(0.0, demand_total - service_capacity) / max(1.0, demand_total)
    backlog_ratio = state["end_backlog"] / max(1.0, demand_total)
    carry_pressure = _clamp(overload_ratio + backlog_ratio, 0.0, 3.0)

    return DayState(
        day=day_plan.day,
        best_satellite=best_sat,
        best_total_cost=float(best_result["total_cost"]),
        best_carbon_objective=float(best_result.get("carbon", 0.0)),
        best_adjusted_total_cost=float(
            best_result.get("adjusted_total_cost", best_result["total_cost"])
        ),
        best_adjusted_carbon_objective=float(
            best_result.get("adjusted_carbon", best_result.get("carbon", 0.0))
        ),
        best_vehicle_count=float(best_result.get("vehicles", 0.0)),
        best_satellite_demand_total=demand_total,
        carry_pressure=carry_pressure,
        cross_day_penalty_total=float(best_result.get("penalty_total", 0.0)),
        hard_violation_count=int(best_result.get("hard_violations", 0)),
        start_depot_inventory=state["start_depot_inventory"],
        start_satellite_inventory=state["start_satellite_inventory"],
        start_backlog=state["start_backlog"],
        input_depot_inbound=state["input_depot_inbound"],
        input_satellite_inbound=state["input_satellite_inbound"],
        transfer_depot_to_satellite=state["transfer_depot_to_satellite"],
        raw_day_demand=state["raw_day_demand"],
        effective_day_demand=state["effective_day_demand"],
        served_today=state["served_today"],
        unserved_today=state["unserved_today"],
        end_depot_inventory=state["end_depot_inventory"],
        end_satellite_inventory=state["end_satellite_inventory"],
        end_backlog=state["end_backlog"],
        inventory_negative_flag=state["inventory_negative_flag"],
        unserved_exists_flag=state["unserved_exists_flag"],
    )


def build_rollup_row(
    day_input: DayInput, day_plan: DayPlan, day_state: DayState
) -> Dict[str, float]:
    return {
        "day": day_input.day,
        "mode": day_plan.mode,
        "customer_start": day_input.customer_start,
        "customer_end": day_input.customer_end,
        "input_cp": day_plan.input_crossover_probability,
        "input_mp": day_plan.input_mutation_probability,
        "effective_cp": day_plan.effective_crossover_probability,
        "effective_mp": day_plan.effective_mutation_probability,
        "prev_best_satellite": (
            day_plan.prev_best_satellite
            if day_plan.prev_best_satellite is not None
            else ""
        ),
        "best_satellite": day_state.best_satellite,
        "best_total_cost": day_state.best_total_cost,
        "best_adjusted_total_cost": day_state.best_adjusted_total_cost,
        "best_carbon": day_state.best_carbon_objective,
        "best_adjusted_carbon": day_state.best_adjusted_carbon_objective,
        "best_vehicles": day_state.best_vehicle_count,
        "best_satellite_demand_total": day_state.best_satellite_demand_total,
        "carry_pressure": day_state.carry_pressure,
        "cross_day_penalty_total": day_state.cross_day_penalty_total,
        "hard_violation_count": day_state.hard_violation_count,
        "start_depot_inventory": day_state.start_depot_inventory,
        "start_satellite_inventory": day_state.start_satellite_inventory,
        "start_backlog": day_state.start_backlog,
        "input_depot_inbound": day_state.input_depot_inbound,
        "input_satellite_inbound": day_state.input_satellite_inbound,
        "transfer_depot_to_satellite": day_state.transfer_depot_to_satellite,
        "raw_day_demand": day_state.raw_day_demand,
        "effective_day_demand": day_state.effective_day_demand,
        "served_today": day_state.served_today,
        "unserved_today": day_state.unserved_today,
        "end_depot_inventory": day_state.end_depot_inventory,
        "end_satellite_inventory": day_state.end_satellite_inventory,
        "end_backlog": day_state.end_backlog,
        "inventory_negative_flag": day_state.inventory_negative_flag,
        "unserved_exists_flag": day_state.unserved_exists_flag,
    }
