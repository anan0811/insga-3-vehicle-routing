"""INSGA3_vrp.py — Self-contained INSGA3 algorithm for 2E-MD-MD-MP-CVRP-SPD.

All utility functions (formerly spread across HNSGA3_vrp, insga3_vrp_load_flow,
insga3_vrp_z2, insga3_vrp_fitness_eval) are inlined here alongside the
INSGA3Algorithm class. Only two external dependencies remain:
  - haversine_python  (spherical distance)
  - FE_gurobipy       (first-echelon Gurobi MILP)
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import openpyxl
from sklearn.cluster import KMeans

from haversine_python import haversine
from FE_gurobipy import FE_gurobi

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_XLSX             = Path(__file__).resolve().parents[2] / "data" / "real_data.xlsx"
NSGA3_MAX_SATELLITES  = 3
INVALID_EMISSION_PENALTY = 1e9
_INSTANCE_ROWS_CACHE  = None

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_instance_rows_from_excel():
    wb = openpyxl.load_workbook(DATA_XLSX, read_only=True, data_only=True)
    try:
        sheet = wb["Sheet1"]
        rows = []
        for row in sheet.iter_rows(values_only=True):
            row_values = [cell for cell in row if cell is not None]
            if row_values:
                rows.append(list(row_values))
    finally:
        wb.close()
    if rows:
        rows = rows[1:]
    return rows


def _get_instance_rows_cache():
    global _INSTANCE_ROWS_CACHE
    if _INSTANCE_ROWS_CACHE is None:
        _INSTANCE_ROWS_CACHE = _load_instance_rows_from_excel()
    return _INSTANCE_ROWS_CACHE


# ── Customer → satellite assignment (K-Means) ─────────────────────────────────

def calculate_satellite_coordinates(instance, number_satellite):
    if not instance:
        return np.empty((0, 2), dtype=float), np.array([], dtype=int)

    features, xy_points = [], []
    for row in instance:
        x = float(row[1])
        y = float(row[2])
        tw_start = float(row[3]) if len(row) > 3 and row[3] is not None else 0.0
        tw_end   = float(row[4]) if len(row) > 4 and row[4] is not None else tw_start
        t_mid    = 0.5 * (tw_start + tw_end)
        features.append([x, y, t_mid])
        xy_points.append([x, y])

    feature_array = np.asarray(features, dtype=float)
    xy_array      = np.asarray(xy_points, dtype=float)

    mean = feature_array.mean(axis=0)
    std  = feature_array.std(axis=0)
    std[std < 1e-12] = 1.0
    scaled = (feature_array - mean) / std

    k = max(1, min(int(number_satellite), len(instance)))
    model          = KMeans(n_clusters=k, n_init="auto", random_state=1)
    customer_group = model.fit_predict(scaled)

    centers = []
    for ci in range(k):
        mask = customer_group == ci
        if np.any(mask):
            centers.append(xy_array[mask].mean(axis=0))
        else:
            c_scaled = model.cluster_centers_[ci]
            centers.append((c_scaled[:2] * std[:2]) + mean[:2])

    return np.asarray(centers, dtype=float), customer_group


# ── Period splitting ──────────────────────────────────────────────────────────

def distinguish_between_periods(instance, periods=720):
    morning_customer, morning_customer_id = [], []
    afternoon_customer, afternoon_customer_id = [], []
    for i in instance:
        if i[4] <= periods:
            morning_customer.append(i)
            morning_customer_id.append([i[0]])
        else:
            afternoon_customer.append(i)
            afternoon_customer_id.append([i[0]])
    return morning_customer, morning_customer_id, afternoon_customer, afternoon_customer_id


# ── Route capacity utilities (from hnsga3_vrp_load_flow) ──────────────────────

_EPS = 1e-9


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
    pickup   = _to_float(row[6]) if len(row) > 6 else 0.0
    return delivery, pickup


def get_aggregated_demand(row: list) -> float:
    delivery, pickup = get_delivery_pickup(row)
    return delivery + pickup


def route_load_profile(
    route: List[object], customer_index: Dict[object, list], capacity: float
) -> dict:
    capacity   = float(capacity)
    start_load = 0.0
    for cid in route:
        row = customer_index.get(cid)
        if row is None:
            continue
        delivery, _ = get_delivery_pickup(row)
        start_load += delivery

    current_load = start_load
    loads    = [current_load]
    max_over  = max(0.0, current_load - capacity)
    max_under = max(0.0, -current_load)

    for cid in route:
        row = customer_index.get(cid)
        if row is None:
            continue
        delivery, pickup = get_delivery_pickup(row)
        current_load = current_load - delivery + pickup
        loads.append(current_load)
        max_over  = max(max_over,  current_load - capacity)
        max_under = max(max_under, -current_load)

    feasible = (max_over <= _EPS) and (max_under <= _EPS)
    return {
        "feasible":   feasible,
        "start_load": start_load,
        "end_load":   current_load,
        "loads":      loads,
        "max_over":   max_over,
        "max_under":  max_under,
    }


def is_route_capacity_feasible(
    route: List[object], customer_index: Dict[object, list], capacity: float
) -> bool:
    if not route:
        return True
    return bool(route_load_profile(route, customer_index, capacity)["feasible"])


def split_sequence_by_capacity(
    sequence: List[object], customer_index: Dict[object, list], capacity: float
) -> List[List[object]]:
    routes: List[List[object]] = []
    current: List[object] = []
    for cid in sequence:
        if not current:
            current = [cid]
            continue
        candidate = current + [cid]
        if is_route_capacity_feasible(candidate, customer_index, capacity):
            current = candidate
        else:
            routes.append(current)
            current = [cid]
    if current:
        routes.append(current)
    return routes


# ── Route decoding (from hnsga3_vrp_z2) ──────────────────────────────────────

def decode_routes_for_z2(
    sequence: Sequence[object],
    instance: Iterable[list],
    vehicle_capacity: float,
) -> List[List[object]]:
    customer_index = build_customer_index(instance)
    return split_sequence_by_capacity(list(sequence), customer_index, vehicle_capacity)


def count_routes_as_vehicles(routes: Sequence[Sequence[object]]) -> int:
    return sum(1 for route in routes if route)


# ── Distance & carbon (MEET model) ────────────────────────────────────────────

def calculate_distance(latitude1, longitude1, latitude2, longitude2):
    return haversine().getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2)


def calculate_time(time_vehicles, instance_row, distance, speed, cost):
    time_vehicles += (distance / speed) * 60
    cost += (
        distance * 0.05
        + 0.2 * max(instance_row[3] - time_vehicles, 0)
        + 0.2 * max(time_vehicles - instance_row[4], 0)
    )
    return time_vehicles, cost


def _safe_speed(speed):
    return max(float(speed), 1e-6)


def carbon_rate_f(speed):
    v = _safe_speed(speed)
    return 110.0 + 0.000375 * (v ** 3) + 8702.0 / v


def load_correction_j(speed):
    v = _safe_speed(speed)
    return 1.27 - 0.00235 * v - 1.33 / v


def calculate_carbon_emission(distance, speed):
    d = float(distance)
    if not math.isfinite(d) or d < 0.0:
        return float("nan"), False
    if d <= 1e-12:
        return 0.0, True
    v = _safe_speed(speed)
    emission = carbon_rate_f(v) * load_correction_j(v) * d / 1000.0
    return emission, emission > 0


def carbon_objective_contribution(distance, speed):
    raw, is_valid = calculate_carbon_emission(distance, speed)
    if is_valid:
        return raw
    return INVALID_EMISSION_PENALTY + abs(raw)


def is_invalid_carbon_objective(value):
    try:
        carbon = float(value)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(carbon):
        return True
    if carbon < 0.0:
        return True
    return carbon >= INVALID_EMISSION_PENALTY


# ── Fitness evaluation (from hnsga3_vrp_fitness_eval) ─────────────────────────

_INFEASIBLE_OBJECTIVE       = 1e12
_ENFORCE_SATELLITE_TW       = False
_SATELLITE_WINDOW_SPAN_MIN  = 1440.0


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
    customer_index      = build_customer_index(instance)
    total_cost          = 0.0
    total_carbon        = 0.0
    num_vehicle         = count_routes_as_vehicles(routes)
    hard_violation_count = 0
    satellite_latest    = None

    if _ENFORCE_SATELLITE_TW:
        for route in routes:
            for cid in route:
                row    = customer_index.get(cid)
                latest = _customer_latest_time(row)
                if latest is None:
                    continue
                if satellite_latest is None or latest > satellite_latest:
                    satellite_latest = latest
        lower = float(time_start) + _SATELLITE_WINDOW_SPAN_MIN
        satellite_latest = max(float(satellite_latest), lower) if satellite_latest else lower

    for route in routes:
        if not route:
            continue
        route_cost   = 0.0
        route_carbon = 0.0
        time_current = time_start

        first = customer_index.get(route[0])
        if first is None:
            continue
        leg = distance_fn(satellite[1], satellite[0], first[2], first[1])
        time_current, route_cost = time_cost_fn(time_current, first, leg, se_vehicle_speed, route_cost)
        route_carbon += carbon_contribution_fn(leg, se_vehicle_speed)

        for prev_id, curr_id in zip(route, route[1:]):
            prev = customer_index.get(prev_id)
            curr = customer_index.get(curr_id)
            if prev is None or curr is None:
                continue
            leg = distance_fn(curr[2], curr[1], prev[2], prev[1])
            time_current, route_cost = time_cost_fn(time_current, curr, leg, se_vehicle_speed, route_cost)
            route_carbon += carbon_contribution_fn(leg, se_vehicle_speed)

        last = customer_index.get(route[-1])
        if last is not None:
            leg = distance_fn(last[2], last[1], satellite[1], satellite[0])
            route_cost   += leg
            route_carbon += carbon_contribution_fn(leg, se_vehicle_speed)
            time_current += (leg / max(float(se_vehicle_speed), 1e-6)) * 60.0

        total_cost   += route_cost
        total_carbon += route_carbon

        if _ENFORCE_SATELLITE_TW and satellite_latest and time_current > satellite_latest + 1e-9:
            hard_violation_count += 1

    if hard_violation_count > 0:
        return [_INFEASIBLE_OBJECTIVE, _INFEASIBLE_OBJECTIVE, _INFEASIBLE_OBJECTIVE]

    return [_safe_round(total_cost, 0), num_vehicle, _safe_round(total_carbon, 6)]


# ── Population utilities ───────────────────────────────────────────────────────

def _fitness_selection_key(fitness):
    if fitness is None or len(fitness) < 3:
        return (1, float("inf"), float("inf"), float("inf"))
    z1, z2, z3 = float(fitness[0]), float(fitness[1]), float(fitness[2])
    infeasible  = int(is_invalid_carbon_objective(z3))
    return (infeasible, z1, z3, z2)


def select_best_population_index(population, front0=None):
    if not population:
        return None
    candidates = [i for i in front0 if 0 <= i < len(population)] if front0 else list(range(len(population)))
    if not candidates:
        candidates = list(range(len(population)))
    return min(candidates, key=lambda i: _fitness_selection_key(population[i][1]))


def find_smae(all_ind, ind):
    for i in all_ind:
        if i[0] == ind:
            return False
    return True


def calculate_satellite_demand(instance, customer_group):
    demand = 0
    for cid in customer_group:
        for row in instance:
            if row[0] == cid:
                if row[5] is not None and row[6] is not None:
                    demand += row[5] + row[6]
    return demand


# ── Route helpers ─────────────────────────────────────────────────────────────

def update_route(ind, se_vehicle_capacity, instance):
    return decode_routes_for_z2(sequence=ind, instance=instance, vehicle_capacity=se_vehicle_capacity)


def _nn_sequence(customer_ids, coord):
    if len(customer_ids) <= 1:
        return list(customer_ids)
    unvisited = list(customer_ids)
    start     = random.choice(unvisited)
    sequence  = [start]
    unvisited.remove(start)
    while unvisited:
        cx, cy  = coord[sequence[-1]]
        nearest = min(unvisited, key=lambda cid: (coord[cid][0] - cx) ** 2 + (coord[cid][1] - cy) ** 2)
        sequence.append(nearest)
        unvisited.remove(nearest)
    return sequence


def satellite_calculate_fitness(ind, se_vehicle_speed, se_vehicle_capacity, instance, time_vehicles, satellite):
    route = update_route(ind, se_vehicle_capacity, instance)
    return evaluate_single_period_fitness(
        routes=route,
        se_vehicle_speed=se_vehicle_speed,
        instance=instance,
        satellite=satellite,
        time_start=time_vehicles,
        distance_fn=calculate_distance,
        time_cost_fn=calculate_time,
        carbon_contribution_fn=carbon_objective_contribution,
        se_vehicle_capacity=se_vehicle_capacity,
    )


# ── VNS neighbourhood operators (used by MOVNS) ──────────────────────────────

def exchange(ind1, se_vehicle_capacity, instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instace)
    if len(route) >= 1:
        r = random.randrange(0, len(route), 1)
        if len(route[r]) > 1:
            a, b = random.sample(range(len(route[r])), 2)
            route[r][a], route[r][b] = route[r][b], route[r][a]
        for i in route:
            update_ind1 += i
        return repair_permutation_sequence(update_ind1, ind1)
    return repair_permutation_sequence(ind, ind1)


def opt2(ind1, se_vehicle_capacity, instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instace)
    if len(route) >= 1:
        r = random.randrange(0, len(route), 1)
        if len(route[r]) > 1:
            a, b = sorted(random.sample(range(len(route[r])), 2))
            re = route[r][:a] + list(reversed(route[r][a : b + 1]))
            if b < len(route[r]) - 1:
                re += route[r][b + 1 :]
        else:
            re = route[r]
        for i in range(len(route)):
            if i == r:
                update_ind1 += re
            else:
                update_ind1 += route[i]
        return repair_permutation_sequence(update_ind1, ind1)
    return repair_permutation_sequence(ind, ind1)


def relocate(ind1, se_vehicle_capacity, instace):
    update_ind1 = []
    asd = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instace)
    if len(route) >= 1:
        r = random.randrange(0, len(route), 1)
        if len(route[r]) > 1:
            a, b = random.sample(range(len(route[r])), 2)
            for i in range(len(route[r])):
                if i == b:
                    continue
                else:
                    asd.append(route[r][i])
            asd.insert(a, route[r][b])
        else:
            asd = route[r]
        for i in range(len(route)):
            if i == r:
                update_ind1 += asd
            else:
                update_ind1 += route[i]
        return repair_permutation_sequence(update_ind1, ind1)
    return repair_permutation_sequence(ind, ind1)


# ── Crossover & mutation ──────────────────────────────────────────────────────

def repair_permutation_sequence(candidate, reference):
    ref_set  = set(reference)
    seen     = set()
    repaired = []
    for gene in candidate:
        if gene in ref_set and gene not in seen:
            repaired.append(gene)
            seen.add(gene)
    for gene in reference:
        if gene not in seen:
            repaired.append(gene)
            seen.add(gene)
    return repaired


def cxOrderedVrp(input_ind1, input_ind2):
    ind1 = copy.deepcopy(input_ind1)
    ind2 = copy.deepcopy(input_ind2)
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2.index(ind2[i])] = False
            holes2[ind1.index(ind1[i])] = False

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[input_ind2.index(temp1[(i + b + 1) % size])]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1
        if not holes2[input_ind1.index(temp2[(i + b + 1) % size])]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    ind1 = repair_permutation_sequence(ind1, input_ind1)
    ind2 = repair_permutation_sequence(ind2, input_ind2)
    return ind1, ind2


def mutation(ind1, mut_prob):
    if len(ind1) > 1 and random.random() <= mut_prob:
        ind2 = copy.deepcopy(ind1)
        if random.random() < 0.5:
            if len(ind2) >= 3:
                pts  = sorted(random.sample(range(len(ind2)), 3))
                a, b = pts[0], pts[2]
            else:
                a, b = sorted(random.sample(range(len(ind2)), 2))
            mutated = ind2[:a] + list(reversed(ind2[a:b + 1]))
            if b < len(ind2) - 1:
                mutated += ind2[b + 1:]
        else:
            a, b = random.sample(range(len(ind2)), 2)
            ind2[a], ind2[b] = ind2[b], ind2[a]
            mutated = ind2
        return repair_permutation_sequence(mutated, ind1)
    return repair_permutation_sequence(ind1, ind1)


# ── NSGA-III sorting & selection ──────────────────────────────────────────────

def dominates(obj_a, obj_b):
    return all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))


def non_dominated_sorting(chroms_obj_record):
    s, n, front, rank = {}, {}, {}, {}
    front[0] = []
    for p in range(len(chroms_obj_record)):
        s[p], n[p] = [], 0
        p_obj = chroms_obj_record[p][1]
        for q in range(len(chroms_obj_record)):
            q_obj = chroms_obj_record[q][1]
            if dominates(p_obj, q_obj):
                if q not in s[p]:
                    s[p].append(q)
            elif dominates(q_obj, p_obj):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i] != []:
        q_front = []
        for p in front[i]:
            for q in s[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in q_front:
                        q_front.append(q)
        i += 1
        front[i] = q_front
    del front[len(front) - 1]
    return front


def _comb(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _get_reference_divisions(num_obj, target_points):
    p = 1
    while _comb(p + num_obj - 1, num_obj - 1) < target_points:
        p += 1
    return p


def _generate_reference_points(num_obj, divisions):
    points = []

    def _recursive(left, total, depth, current):
        if depth == num_obj - 1:
            current.append(left / total)
            points.append(np.array(current, dtype=float))
            current.pop()
            return
        for i in range(left + 1):
            current.append(i / total)
            _recursive(left - i, total, depth + 1, current)
            current.pop()

    _recursive(divisions, divisions, 0, [])
    return np.array(points, dtype=float)


def _normalize_for_nsga3(obj_matrix):
    ideal     = np.min(obj_matrix, axis=0)
    translated = obj_matrix - ideal
    if np.allclose(translated, 0):
        return translated
    m = translated.shape[1]
    extreme_idx = []
    for i in range(m):
        weights = np.full(m, 1e-6)
        weights[i] = 1.0
        asf = np.max(translated / weights, axis=1)
        extreme_idx.append(int(np.argmin(asf)))
    extreme_points = translated[extreme_idx, :]
    intercepts = None
    try:
        if np.linalg.matrix_rank(extreme_points) == m:
            plane = np.linalg.solve(extreme_points, np.ones(m))
            if np.all(plane > 1e-12):
                intercepts = 1.0 / plane
    except np.linalg.LinAlgError:
        intercepts = None
    if intercepts is None or np.any(intercepts <= 1e-12):
        intercepts = np.max(translated, axis=0)
    intercepts[intercepts <= 1e-12] = 1.0
    return translated / intercepts


def _associate_to_reference_points(indices, normalized_obj, reference_points):
    ref_norm = np.linalg.norm(reference_points, axis=1)
    ref_norm[ref_norm == 0] = 1.0
    association, dist_to_ref = {}, {}
    for row_idx, ind in enumerate(indices):
        y      = normalized_obj[row_idx]
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            nearest = int(np.argmin(np.linalg.norm(reference_points, axis=1)))
            association[ind]  = nearest
            dist_to_ref[ind]  = 0.0
            continue
        perp_distances = []
        for rp_idx, rp in enumerate(reference_points):
            proj_len = np.dot(y, rp) / ref_norm[rp_idx]
            proj     = (proj_len / ref_norm[rp_idx]) * rp
            perp_distances.append(np.linalg.norm(y - proj))
        nearest = int(np.argmin(perp_distances))
        association[ind] = nearest
        dist_to_ref[ind] = float(perp_distances[nearest])
    return association, dist_to_ref


def _nsga3_niching(last_front, selected_indices, chroms_obj_record, n_select, population_size):
    if n_select <= 0 or not last_front:
        return []
    num_obj     = len(chroms_obj_record[last_front[0]][1])
    divisions   = _get_reference_divisions(num_obj, max(population_size, len(last_front)))
    ref_points  = _generate_reference_points(num_obj, divisions)
    union       = selected_indices + list(last_front)
    union_obj   = np.array([chroms_obj_record[i][1] for i in union], dtype=float)
    normalized  = _normalize_for_nsga3(union_obj)
    assoc, dist = _associate_to_reference_points(union, normalized, ref_points)

    niche_counts = [0] * len(ref_points)
    for idx in selected_indices:
        niche_counts[assoc[idx]] += 1

    ref_to_candidates = {i: [] for i in range(len(ref_points))}
    for idx in last_front:
        ref_to_candidates[assoc[idx]].append(idx)

    chosen, remaining = [], set(last_front)
    while len(chosen) < n_select and remaining:
        active_refs = [rp for rp, cands in ref_to_candidates.items() if any(c in remaining for c in cands)]
        if not active_refs:
            break
        min_count      = min(niche_counts[rp] for rp in active_refs)
        candidate_refs = [rp for rp in active_refs if niche_counts[rp] == min_count]
        selected_ref   = random.choice(candidate_refs)
        candidates     = [c for c in ref_to_candidates[selected_ref] if c in remaining]
        if not candidates:
            niche_counts[selected_ref] = 10 ** 9
            continue
        chosen_idx = min(candidates, key=lambda c: dist[c]) if niche_counts[selected_ref] == 0 else random.choice(candidates)
        chosen.append(chosen_idx)
        remaining.remove(chosen_idx)
        niche_counts[selected_ref] += 1

    if len(chosen) < n_select:
        leftovers = [i for i in last_front if i not in chosen]
        random.shuffle(leftovers)
        chosen.extend(leftovers[:n_select - len(chosen)])
    return chosen


def selection(population_size, front, chroms_obj_record, best=False):
    if best:
        return [chroms_obj_record[i] for i in front]
    selected, front_idx = [], 0
    while front_idx < len(front) and len(selected) + len(front[front_idx]) <= population_size:
        selected.extend(front[front_idx])
        front_idx += 1
    if front_idx < len(front) and len(selected) < population_size:
        remain = population_size - len(selected)
        chosen = _nsga3_niching(front[front_idx], selected, chroms_obj_record, remain, population_size)
        selected.extend(chosen)
    return [chroms_obj_record[i] for i in selected], selected


# ── K-Medoids (PAM) ───────────────────────────────────────────────────────────

def _kmedoids_spatial(ids, coord, k, max_iter=50):
    n = len(ids)
    k = max(1, min(k, n))
    if k == 1:
        return [list(ids)]

    def dist(a, b):
        ax, ay = coord[a]
        bx, by = coord[b]
        return math.hypot(ax - bx, ay - by)

    medoids = random.sample(ids, k)
    for _ in range(max_iter):
        clusters = {m: [] for m in medoids}
        for cid in ids:
            nearest = min(medoids, key=lambda m: dist(cid, m))
            clusters[nearest].append(cid)
        new_medoids = []
        for m, members in clusters.items():
            if not members:
                new_medoids.append(m)
                continue
            best = min(members, key=lambda c: sum(dist(c, o) for o in members))
            new_medoids.append(best)
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids

    clusters = {m: [] for m in medoids}
    for cid in ids:
        nearest = min(medoids, key=lambda m: dist(cid, m))
        clusters[nearest].append(cid)
    return [v for v in clusters.values() if v]


def _kmedoids_from_matrix(ids, dist_matrix, k, max_iter=50):
    n = len(ids)
    k = max(1, min(k, n))
    if k == 1:
        return [list(ids)]

    idx_all    = list(range(n))
    medoid_idx = random.sample(idx_all, k)

    for _ in range(max_iter):
        clusters = {m: [] for m in medoid_idx}
        for i in idx_all:
            nearest = min(medoid_idx, key=lambda m: dist_matrix[i][m])
            clusters[nearest].append(i)
        new_medoids = []
        for m, members in clusters.items():
            if not members:
                new_medoids.append(m)
                continue
            best = min(members, key=lambda c: sum(dist_matrix[c][o] for o in members))
            new_medoids.append(best)
        if set(new_medoids) == set(medoid_idx):
            break
        medoid_idx = new_medoids

    clusters = {m: [] for m in medoid_idx}
    for i in idx_all:
        nearest = min(medoid_idx, key=lambda m: dist_matrix[i][m])
        clusters[nearest].append(i)
    return [[ids[i] for i in v] for v in clusters.values() if v]


# ── INSGA3 Algorithm ──────────────────────────────────────────────────────────

class INSGA3Algorithm:
    """INSGA3: Triple-Initialization NSGA-III for 2E-MD-MD-MP-CVRP-SPD.

    Initialization strategy (pop_size evenly split):
      1. Logistic Chaos Map       — diverse permutations via x_{n+1}=r*x_n*(1-x_n)
      2. K-Medoids (spatial)      — geographically compact cluster-then-route
      3. Temporal-Spatial K-Medoids — cluster by spatial + time-window distance
    Shortfall slots filled by additional Chaos Map individuals.

    Evolutionary loop: OX crossover + inversion mutation + NSGA-III niching.
    No local search.
    """

    def __init__(self):
        self.start_customer_number = 0
        self.end_customer_number   = 99
        self.crossover_probability = 0.85
        self.mut_prob              = 0.1
        self.num_gen               = 1000
        self.pop_size              = 30
        self.ts_alpha              = 0.5
        self.chaos_r               = 3.9
        self.fe_vehicle_capacity   = 4000
        self.se_vehicle_capacity   = 400
        self.se_vehicle_speed      = 50
        self.depot = [
            ["d1", 120.373731, 36.185609],
            ["d2", 118.054927, 36.813487],
            ["d3", 116.897877, 36.611274],
        ]
        self.number_satellite = 2
        self.centers          = None
        self.customer_group   = None
        self.json_instance    = []

        self.all_morning_customer_id_and_fitness1   = []
        self.all_afternoon_customer_id_and_fitness1 = []
        self.all_morning_customer_id_and_fitness2   = []
        self.all_afternoon_customer_id_and_fitness2 = []
        self.all_morning_customer_id_and_fitness3   = []
        self.all_afternoon_customer_id_and_fitness3 = []

        self.start_time          = time.time()
        self.all_number_vehicles = 0
        self.all_time            = 0
        self.FE_all_cost         = 0
        self.SE_all_cost         = 0
        self.SE_all_carbon       = 0
        self.satellite_demand    = []

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_instance(self):
        all_rows = _get_instance_rows_cache()
        subset   = all_rows[self.start_customer_number: self.end_customer_number + 1]
        self.json_instance = [row[:] for row in subset]
        self.centers, self.customer_group = calculate_satellite_coordinates(
            self.json_instance, self.number_satellite
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _coord_map(self, flat_ids):
        id_set = set(flat_ids)
        return {
            row[0]: (float(row[1]), float(row[2]))
            for row in self.json_instance
            if row[0] in id_set
        }

    def _evaluate(self, candidate, sat_idx, time_start):
        return satellite_calculate_fitness(
            candidate,
            self.se_vehicle_speed,
            self.se_vehicle_capacity,
            self.json_instance,
            time_vehicles=time_start,
            satellite=self.centers[sat_idx],
        )

    def _capacity_k(self, flat_ids):
        cust_index   = {row[0]: row for row in self.json_instance}
        total_demand = 0.0
        for cid in flat_ids:
            row = cust_index.get(cid)
            if row is None:
                continue
            delivery = float(row[5]) if len(row) > 5 else 0.0
            pickup   = float(row[6]) if len(row) > 6 else 0.0
            total_demand += delivery + pickup
        return max(2, min(math.ceil(total_demand / self.se_vehicle_capacity), len(flat_ids)))

    def _build_chromosome_from_clusters(self, clusters, sat_idx, coord):
        sat_lon = float(self.centers[sat_idx][0])
        sat_lat = float(self.centers[sat_idx][1])

        def centroid_dist(cluster):
            cx = sum(coord[c][0] for c in cluster) / len(cluster)
            cy = sum(coord[c][1] for c in cluster) / len(cluster)
            return math.hypot(cx - sat_lon, cy - sat_lat)

        sorted_clusters = sorted(clusters, key=centroid_dist)
        n_c = len(sorted_clusters)
        rot = random.randint(0, n_c - 1)
        ordered = sorted_clusters[rot:] + sorted_clusters[:rot]

        chromosome = []
        for cluster in ordered:
            if len(cluster) == 1:
                chromosome.extend(cluster)
            else:
                chromosome.extend(_nn_sequence(cluster, coord))
        return chromosome

    # ── Initialization Method 1: Logistic Chaos Map ───────────────────────────

    def _make_chaos_population(self, flat_ids, sat_idx, time_start, n_slots):
        pop = []
        n   = len(flat_ids)
        if not n:
            return pop
        attempts = 0
        while len(pop) < n_slots and attempts < n_slots * 30:
            attempts += 1
            x      = random.random()
            values = []
            for _ in range(n):
                x = self.chaos_r * x * (1.0 - x)
                values.append(x)
            order     = sorted(range(n), key=lambda i: values[i])
            candidate = [flat_ids[i] for i in order]
            if find_smae(pop, candidate):
                pop.append([candidate, self._evaluate(candidate, sat_idx, time_start)])
        return pop

    # ── Initialization Method 2: K-Medoids (spatial) ─────────────────────────

    def _make_kmedoids_population(self, flat_ids, sat_idx, time_start, n_slots):
        pop  = []
        base = list(flat_ids)
        if not base:
            return pop
        coord    = self._coord_map(base)
        k        = self._capacity_k(base)
        clusters = _kmedoids_spatial(base, coord, k)
        attempts = 0
        while len(pop) < n_slots and attempts < n_slots * 30:
            attempts += 1
            candidate = self._build_chromosome_from_clusters(clusters, sat_idx, coord)
            if find_smae(pop, candidate):
                pop.append([candidate, self._evaluate(candidate, sat_idx, time_start)])
        return pop

    # ── Initialization Method 3: Temporal-Spatial K-Medoids ──────────────────

    def _temporal_spatial_dist_matrix(self, flat_ids):
        n          = len(flat_ids)
        id_to_row  = {row[0]: row for row in self.json_instance if row[0] in set(flat_ids)}
        spatial    = np.zeros((n, n))
        temporal   = np.zeros((n, n))
        for i in range(n):
            ri       = id_to_row[flat_ids[i]]
            tw_mid_i = (float(ri[3]) + float(ri[4])) / 2.0
            for j in range(n):
                if i == j:
                    continue
                rj       = id_to_row[flat_ids[j]]
                tw_mid_j = (float(rj[3]) + float(rj[4])) / 2.0
                spatial[i][j]  = haversine().getDistanceBetweenPointsNew(
                    float(ri[1]), float(ri[2]), float(rj[1]), float(rj[2])
                )
                temporal[i][j] = abs(tw_mid_i - tw_mid_j)
        s_max = spatial.max()  if spatial.max()  > 0 else 1.0
        t_max = temporal.max() if temporal.max() > 0 else 1.0
        return self.ts_alpha * (spatial / s_max) + (1 - self.ts_alpha) * (temporal / t_max)

    def _make_temporal_spatial_population(self, flat_ids, sat_idx, time_start, n_slots):
        pop  = []
        base = list(flat_ids)
        if not base:
            return pop
        coord       = self._coord_map(base)
        k           = self._capacity_k(base)
        dist_matrix = self._temporal_spatial_dist_matrix(base)
        clusters    = _kmedoids_from_matrix(base, dist_matrix, k)
        attempts    = 0
        while len(pop) < n_slots and attempts < n_slots * 30:
            attempts += 1
            candidate = self._build_chromosome_from_clusters(clusters, sat_idx, coord)
            if find_smae(pop, candidate):
                pop.append([candidate, self._evaluate(candidate, sat_idx, time_start)])
        return pop

    # ── Combined initialization ────────────────────────────────────────────────

    def initial_population(self):
        chaos_slots = self.pop_size // 3
        km_slots    = self.pop_size // 3
        ts_slots    = self.pop_size - chaos_slots - km_slots

        for sat_idx in range(self.number_satellite):
            group = [
                self.json_instance[j]
                for j in range(len(self.customer_group))
                if int(self.customer_group[j]) == sat_idx
            ]
            _, morning_id_lists, _, afternoon_id_lists = distinguish_between_periods(group)
            morning_flat   = [lst[0] for lst in morning_id_lists]
            afternoon_flat = [lst[0] for lst in afternoon_id_lists]

            attr_m = f"all_morning_customer_id_and_fitness{sat_idx + 1}"
            attr_a = f"all_afternoon_customer_id_and_fitness{sat_idx + 1}"

            for flat, time_start, attr in (
                (morning_flat,   540, attr_m),
                (afternoon_flat, 720, attr_a),
            ):
                if not flat:
                    continue
                chaos_pop = self._make_chaos_population(flat, sat_idx, time_start, chaos_slots)
                km_pop    = self._make_kmedoids_population(flat, sat_idx, time_start, km_slots)
                ts_pop    = self._make_temporal_spatial_population(flat, sat_idx, time_start, ts_slots)

                merged = chaos_pop[:]
                for ind in km_pop:
                    if find_smae(merged, ind[0]):
                        merged.append(ind)
                for ind in ts_pop:
                    if find_smae(merged, ind[0]):
                        merged.append(ind)

                if len(merged) < self.pop_size:
                    extra = self._make_chaos_population(flat, sat_idx, time_start, self.pop_size - len(merged))
                    for ind in extra:
                        if find_smae(merged, ind[0]):
                            merged.append(ind)

                if not merged:
                    candidate = flat[:]
                    merged.append([candidate, self._evaluate(candidate, sat_idx, time_start)])

                setattr(self, attr, merged)

    # ── Evolutionary loop ─────────────────────────────────────────────────────

    def _evolve_population(self, pop, sat_idx, time_start):
        if not pop or len(pop[0][0]) <= 1:
            return pop
        offspring = []
        if self.crossover_probability >= random.random():
            indices = list(range(len(pop)))
            random.shuffle(indices)
            for k in range(0, len(indices) - 1, 2):
                i, j = indices[k], indices[k + 1]
                child1, child2 = cxOrderedVrp(pop[i][0], pop[j][0])
                for child in (child1, child2):
                    if find_smae(pop + offspring, child):
                        offspring.append([child, self._evaluate(child, sat_idx, time_start)])
        for ind in pop:
            child = mutation(ind[0], self.mut_prob)
            if find_smae(pop + offspring, child):
                offspring.append([child, self._evaluate(child, sat_idx, time_start)])
        combined = pop + offspring
        front    = non_dominated_sorting(combined)
        trimmed, _ = selection(self.pop_size, front, combined)
        return trimmed

    def runGenerations(self):
        configs = [
            ("all_morning_customer_id_and_fitness1",   0, 540),
            ("all_morning_customer_id_and_fitness2",   1, 540),
            ("all_morning_customer_id_and_fitness3",   2, 540),
            ("all_afternoon_customer_id_and_fitness1", 0, 720),
            ("all_afternoon_customer_id_and_fitness2", 1, 720),
            ("all_afternoon_customer_id_and_fitness3", 2, 720),
        ]
        for _gen in range(self.num_gen):
            for attr, sat_idx, time_start in configs:
                if sat_idx >= self.number_satellite:
                    continue
                pop = getattr(self, attr)
                if pop:
                    setattr(self, attr, self._evolve_population(pop, sat_idx, time_start))

    # ── Result aggregation ────────────────────────────────────────────────────

    def _select_best_solution(self, pop):
        front    = non_dominated_sorting(pop)
        front0   = front.get(0, [])
        best_idx = select_best_population_index(pop, front0)
        if best_idx is None:
            raise ValueError("Cannot select best solution from empty population.")
        return pop[best_idx]

    def _safe_select(self, pop):
        return self._select_best_solution(pop) if pop else None

    def result(self):
        self.satellite_demand = []

        best1m = self._safe_select(self.all_morning_customer_id_and_fitness1)
        best1a = self._safe_select(self.all_afternoon_customer_id_and_fitness1)
        sat1_demand = 0
        if best1m:
            sat1_demand += calculate_satellite_demand(self.json_instance, best1m[0])
        if best1a:
            sat1_demand += calculate_satellite_demand(self.json_instance, best1a[0])
        self.satellite_demand.append(sat1_demand)

        best2m = best2a = best3m = best3a = None
        if self.number_satellite >= 2:
            best2m = self._safe_select(self.all_morning_customer_id_and_fitness2)
            best2a = self._safe_select(self.all_afternoon_customer_id_and_fitness2)
            sat2_demand = 0
            if best2m:
                sat2_demand += calculate_satellite_demand(self.json_instance, best2m[0])
            if best2a:
                sat2_demand += calculate_satellite_demand(self.json_instance, best2a[0])
            self.satellite_demand.append(sat2_demand)

        if self.number_satellite >= 3:
            best3m = self._safe_select(self.all_morning_customer_id_and_fitness3)
            best3a = self._safe_select(self.all_afternoon_customer_id_and_fitness3)
            sat3_demand = 0
            if best3m:
                sat3_demand += calculate_satellite_demand(self.json_instance, best3m[0])
            if best3a:
                sat3_demand += calculate_satellite_demand(self.json_instance, best3a[0])
            self.satellite_demand.append(sat3_demand)

        candidates = [best1m, best1a, best2m, best2a, best3m, best3a]
        pops = [p for p in candidates if p is not None]

        self.all_number_vehicles = sum(p[1][1] for p in pops)
        self.SE_all_cost         = sum(p[1][0] for p in pops)
        self.SE_all_carbon       = sum(p[1][2] for p in pops)
        self.SE_all_cost        += self.all_number_vehicles * 50 + self.number_satellite * 500

    # ── First echelon (Gurobi) ────────────────────────────────────────────────

    def First_route(self):
        gurobi_model  = FE_gurobi()
        self.FE_all_cost = gurobi_model.main(
            self.depot,
            self.number_satellite,
            self.centers,
            list(range(self.number_satellite)),
            self.satellite_demand,
            self.fe_vehicle_capacity,
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def runMain(self):
        self.number_satellite = max(1, min(int(self.number_satellite), NSGA3_MAX_SATELLITES))
        self.load_instance()
        self.initial_population()
        self.runGenerations()
        self.result()
        self.First_route()
        self.all_time = time.time() - self.start_time
