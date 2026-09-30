"""pure_nsga2.py — Standard NSGA-II for 2E-MD-MD-MP-CVRP-SPD.

Differences from HNSGA-III (HNSGA3_vrp.py):
  - Random initialisation  (no NN heuristic)
  - No local search        (no VNS: exchange / 2-opt / relocate)
  - Pure NSGA-II loop:     OX crossover + inversion mutation + crowding-distance selection
  - FE routing via Gurobi  (same as HNSGA-III — required by problem structure)

Differences from nsga3.py (standard NSGA-III):
  - Selection: crowding distance  (NSGA-II)  instead of reference-point niching (NSGA-III)
"""

import random
import time
import sys
from pathlib import Path

_INSGA3_DIR = Path(__file__).resolve().parent.parent / "INSGA-III"
if str(_INSGA3_DIR) not in sys.path:
    sys.path.insert(0, str(_INSGA3_DIR))

from INSGA3_vrp import (
    _get_instance_rows_cache,
    calculate_satellite_coordinates,
    distinguish_between_periods,
    satellite_calculate_fitness,
    cxOrderedVrp,
    non_dominated_sorting,
    select_best_population_index,
    calculate_satellite_demand,
    find_smae,
    repair_permutation_sequence,
    NSGA3_MAX_SATELLITES,
)
from FE_gurobipy import FE_gurobi


# ── Standard inversion mutation (NSGA-II) ────────────────────────────────────

def _inversion_mutation(ind, mut_prob):
    """Standard inversion mutation: randomly reverse a sub-sequence.

    Selects two positions a < b at random and reverses ind[a:b+1].
    Matches JHENG (2025) and the classic NSGA-II mutation operator.
    """
    if len(ind) > 1 and random.random() <= mut_prob:
        a, b = sorted(random.sample(range(len(ind)), 2))
        mutated = ind[:a] + list(reversed(ind[a:b + 1])) + ind[b + 1:]
        return repair_permutation_sequence(mutated, ind)
    return ind[:]


# ── Crowding distance (NSGA-II selection) ────────────────────────────────────

def _crowding_distance(front_indices, population):
    """Compute crowding distance for all individuals in one Pareto front."""
    n = len(front_indices)
    if n == 0:
        return {}
    cd = {idx: 0.0 for idx in front_indices}
    if n == 1:
        cd[front_indices[0]] = float("inf")
        return cd

    num_obj = len(population[front_indices[0]][1])
    for m in range(num_obj):
        sorted_front = sorted(front_indices, key=lambda i: population[i][1][m])
        cd[sorted_front[0]]  = float("inf")
        cd[sorted_front[-1]] = float("inf")
        f_min = population[sorted_front[0]][1][m]
        f_max = population[sorted_front[-1]][1][m]
        denom = f_max - f_min if f_max > f_min else 1.0
        for k in range(1, n - 1):
            cd[sorted_front[k]] += (
                population[sorted_front[k + 1]][1][m]
                - population[sorted_front[k - 1]][1][m]
            ) / denom
    return cd


def _nsga2_select(pop_size, front_dict, population):
    """NSGA-II environmental selection using crowding distance.

    Returns (new_population_list, selected_indices).
    Mirrors the signature of NSGA-III selection() so it can be used
    as a drop-in replacement.
    """
    selected = []
    front_idx = 0
    max_front = max(front_dict.keys()) if front_dict else -1

    while front_idx <= max_front:
        current_front = front_dict.get(front_idx, [])
        if len(selected) + len(current_front) <= pop_size:
            selected.extend(current_front)
            front_idx += 1
        else:
            # Last partial front: sort by crowding distance descending
            remain = pop_size - len(selected)
            cd = _crowding_distance(current_front, population)
            sorted_front = sorted(current_front, key=lambda i: cd.get(i, 0.0), reverse=True)
            selected.extend(sorted_front[:remain])
            break

    population_list = [population[idx] for idx in selected]
    return population_list, selected


# ── Pure NSGA-II Algorithm ────────────────────────────────────────────────────

class PureNSGA2Algorithm:
    """Standard NSGA-II applied to 2E-MD-MD-MP-CVRP-SPD.

    No NN initialisation, no VNS local search.
    Only difference from PureNSGA3 (nsga3.py): crowding-distance selection
    instead of NSGA-III reference-point niching.
    """

    def __init__(self):
        self.start_customer_number = 0
        self.end_customer_number   = 99
        self.crossover_probability = 0.85
        self.mut_prob              = 0.5
        self.num_gen               = 1000
        self.pop_size              = 30
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

        self.start_time        = time.time()
        self.all_number_vehicles = 0
        self.all_time          = 0
        self.FE_all_cost       = 0
        self.SE_all_cost       = 0
        self.SE_all_carbon     = 0
        self.satellite_demand  = []

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_instance(self):
        all_rows = _get_instance_rows_cache()
        subset = all_rows[self.start_customer_number: self.end_customer_number + 1]
        self.json_instance = [row[:] for row in subset]
        self.centers, self.customer_group = calculate_satellite_coordinates(
            self.json_instance, self.number_satellite
        )

    # ── Random initialisation ─────────────────────────────────────────────────

    def _make_random_population(self, flat_ids, sat_idx, time_start):
        pop = []
        base = list(flat_ids)
        if not base:
            return pop

        attempts = 0
        while len(pop) < self.pop_size and attempts < self.pop_size * 20:
            attempts += 1
            candidate = base[:]
            random.shuffle(candidate)
            if find_smae(pop, candidate):
                fitness = satellite_calculate_fitness(
                    candidate,
                    self.se_vehicle_speed,
                    self.se_vehicle_capacity,
                    self.json_instance,
                    time_vehicles=time_start,
                    satellite=self.centers[sat_idx],
                )
                pop.append([candidate, fitness])

        if not pop:
            fitness = satellite_calculate_fitness(
                base,
                self.se_vehicle_speed,
                self.se_vehicle_capacity,
                self.json_instance,
                time_vehicles=time_start,
                satellite=self.centers[sat_idx],
            )
            pop.append([base, fitness])

        return pop

    def initial_population(self):
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

            if morning_flat:
                setattr(self, attr_m, self._make_random_population(morning_flat, sat_idx, 540))
            if afternoon_flat:
                setattr(self, attr_a, self._make_random_population(afternoon_flat, sat_idx, 720))

    # ── Evolutionary loop (crossover + mutation, NSGA-II crowding selection) ──

    def _evolve_population(self, pop, sat_idx, time_start):
        """One generation: OX crossover + standard inversion mutation + NSGA-II crowding selection."""
        if not pop or len(pop[0][0]) <= 1:
            return pop

        offspring = []

        # Ordered crossover
        if self.crossover_probability >= random.random():
            indices = list(range(len(pop)))
            random.shuffle(indices)
            for k in range(0, len(indices) - 1, 2):
                i, j = indices[k], indices[k + 1]
                child1, child2 = cxOrderedVrp(pop[i][0], pop[j][0])
                for child in (child1, child2):
                    if find_smae(pop + offspring, child):
                        fit = satellite_calculate_fitness(
                            child,
                            self.se_vehicle_speed,
                            self.se_vehicle_capacity,
                            self.json_instance,
                            time_vehicles=time_start,
                            satellite=self.centers[sat_idx],
                        )
                        offspring.append([child, fit])

        # Standard inversion mutation (NSGA-II)
        for ind in pop:
            child = _inversion_mutation(ind[0], self.mut_prob)
            if find_smae(pop + offspring, child):
                fit = satellite_calculate_fitness(
                    child,
                    self.se_vehicle_speed,
                    self.se_vehicle_capacity,
                    self.json_instance,
                    time_vehicles=time_start,
                    satellite=self.centers[sat_idx],
                )
                offspring.append([child, fit])

        combined = pop + offspring

        # NSGA-II crowding-distance selection
        front = non_dominated_sorting(combined)
        trimmed, _ = _nsga2_select(self.pop_size, front, combined)
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
        front = non_dominated_sorting(pop)
        front0 = front.get(0, [])
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

        self.SE_all_cost += self.all_number_vehicles * 50 + self.number_satellite * 500

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
