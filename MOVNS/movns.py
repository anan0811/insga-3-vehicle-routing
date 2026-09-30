"""movns.py — Multi-Objective Variable Neighborhood Search for 2E-MD-MD-MP-CVRP-SPD.

Single-solution VNS (structure follows JHENG's MOVNS_vrp.py):
  - Initialization : one random solution per block
  - VNS loop       : per block per generation, try 5 operators against the
                     current single solution; accept if Pareto-dominates;
                     reset restart counter i=0 on improvement (then i += 1)
  - Neighborhoods  : exchange / opt2 / relocate / s_exchange / all_s_exchange
  - Three objectives: cost, NV, carbon  (extends JHENG's 2-objective version)
  - VNS_method = 3  (max rounds without improvement before moving on)
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

_INSGA3_DIR = Path(__file__).resolve().parent.parent / "INSGA-III"
if str(_INSGA3_DIR) not in sys.path:
    sys.path.insert(0, str(_INSGA3_DIR))

from INSGA3_vrp import (
    _get_instance_rows_cache,
    calculate_satellite_coordinates,
    distinguish_between_periods,
    satellite_calculate_fitness,
    calculate_satellite_demand,
    opt2,
    exchange,
    relocate,
    NSGA3_MAX_SATELLITES,
)
from FE_gurobipy import FE_gurobi


# ── Extra neighbourhood operators (from JHENG's MOVNS_vrp.py) ────────────────

def _update_route(ind, se_vehicle_capacity, instance):
    capacity = 0
    total_route = []
    route = []
    for cid in ind:
        for row in instance:
            if row[0] == cid:
                capacity += row[5] + row[6]
                if capacity >= se_vehicle_capacity:
                    total_route.append(route)
                    capacity = 0
                    route = [cid]
                else:
                    route.append(cid)
                break
    if route:
        total_route.append(route)
    return total_route


def s_exchange(ind, se_vehicle_capacity, instance):
    """Swap one customer between two different routes."""
    routes = _update_route(copy.deepcopy(ind), se_vehicle_capacity, instance)
    if len(routes) > 1:
        r1, r2 = sorted(random.sample(range(len(routes)), 2))
        if routes[r1] and routes[r2]:
            a = random.randrange(len(routes[r1]))
            b = random.randrange(len(routes[r2]))
            routes[r1][a], routes[r2][b] = routes[r2][b], routes[r1][a]
    flat = [c for r in routes for c in r]
    return flat if flat else ind


def all_s_exchange(ind, se_vehicle_capacity, instance):
    """Swap two entire routes."""
    routes = _update_route(copy.deepcopy(ind), se_vehicle_capacity, instance)
    if len(routes) > 1:
        r1, r2 = sorted(random.sample(range(len(routes)), 2))
        routes[r1], routes[r2] = routes[r2], routes[r1]
    flat = [c for r in routes for c in r]
    return flat if flat else ind


# ── Pareto dominance (3 objectives, all minimise) ─────────────────────────────

def _dominates(fit_a, fit_b):
    a0, a1, a2 = fit_a[0], fit_a[1], fit_a[2]
    b0, b1, b2 = fit_b[0], fit_b[1], fit_b[2]
    return (a0 <= b0 and a1 <= b1 and a2 <= b2) and (a0 < b0 or a1 < b1 or a2 < b2)


# ── Algorithm ────────────────────────────────────────────────────────────────

class MOVNSAlgorithm:
    """Single-solution MOVNS applied to 2E-MD-MD-MP-CVRP-SPD.

    One random solution is maintained per block (satellite × period).
    Each generation applies 5 VNS operators sequentially; a neighbour is
    accepted when it Pareto-dominates the current solution.  The restart
    counter resets to 1 on improvement, mirroring JHENG's implementation.
    """

    def __init__(self):
        self.start_customer_number = 0
        self.end_customer_number   = 99
        self.crossover_probability = 0.65   # kept for interface compatibility
        self.mut_prob              = 0.60   # kept for interface compatibility
        self.num_gen               = 1000
        self.pop_size              = 30     # kept for interface compatibility
        self.VNS_method            = 3
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
        subset = all_rows[self.start_customer_number: self.end_customer_number + 1]
        self.json_instance = [row[:] for row in subset]
        self.centers, self.customer_group = calculate_satellite_coordinates(
            self.json_instance, self.number_satellite
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self, candidate, sat_idx, time_start):
        return satellite_calculate_fitness(
            candidate,
            self.se_vehicle_speed,
            self.se_vehicle_capacity,
            self.json_instance,
            time_vehicles=time_start,
            satellite=self.centers[sat_idx],
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def initial_population(self):
        """One random solution per block (satellite × period)."""
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
                if flat:
                    candidate = flat[:]
                    random.shuffle(candidate)
                    setattr(self, attr, [[candidate, self._evaluate(candidate, sat_idx, time_start)]])

    # ── Single-solution VNS step ──────────────────────────────────────────────

    def _vns_step(self, pop, sat_idx, time_start):
        """Apply 5 VNS operators to the single current solution.

        Mirrors JHENG's while-loop structure:
          - capture base fitness once per while-iteration
          - try all 5 operators; accept each that Pareto-dominates base
          - if any accepted: i = 0  (then i += 1 → i = 1, loop continues)
          - after all operators: i += 1
          - loop exits when i > VNS_method
        """
        if not pop:
            return pop

        current_ind, current_fit = pop[0][0], pop[0][1]
        cap = self.se_vehicle_capacity
        inst = self.json_instance

        i = 1
        while i <= self.VNS_method:
            base_fit = current_fit          # snapshot for this round

            for fn in (exchange, opt2, relocate, s_exchange, all_s_exchange):
                neighbor     = fn(current_ind, cap, inst)
                neighbor_fit = self._evaluate(neighbor, sat_idx, time_start)
                if _dominates(neighbor_fit, base_fit):
                    current_ind, current_fit = neighbor, neighbor_fit
                    i = 0                   # reset counter (same as JHENG)

            i += 1

        return [[current_ind, current_fit]]

    # ── Main VNS loop ─────────────────────────────────────────────────────────

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
                    setattr(self, attr, self._vns_step(pop, sat_idx, time_start))

    # ── Result aggregation ────────────────────────────────────────────────────

    def _safe_select(self, pop):
        return pop[0] if pop else None

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
        gurobi_model = FE_gurobi()
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
