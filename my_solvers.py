"""
Implementations of Knapsack solvers for COMA project.

Each function must adhere to the standard solver interface:
fit(instance: Dict, timeout: Optional[float] = None, seed: Optional[int] = None) -> Tuple[List[int], float, Dict]

Returns:
1.  best_solution (List[int]): List of item IDs included in the knapsack.
2.  best_value (float): Total value of the best_solution.
3.  logs (Dict): A dictionary for extra data (e.g., final_weight, convergence_history).
"""

import time
import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import heapq
import sys
import logging
import neal
import dimod

# --- compatibility decorator: accept instance=... or items+capacity ---
from functools import wraps

def accept_instance(func):
    """
    Decorator that allows calling solver(instance=inst, ...) where inst is a dict
    with keys 'items' and 'capacity'. If 'instance' is present it will set
    kwargs['items'] and kwargs['capacity'] before calling the wrapped function.
    Works with existing signatures that accept items, capacity as keyword args.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        inst = kwargs.pop('instance', None)
        if inst is not None:
            # only set items/capacity if not already provided explicitly
            if 'items' not in kwargs:
                kwargs['items'] = inst.get('items')
            if 'capacity' not in kwargs:
                kwargs['capacity'] = inst.get('capacity')
        return func(*args, **kwargs)
    return wrapper
# ----------------------------------------------------------------------

def _base_logs(message, runtime, final_value=None, final_weight=None,
               solution_size=None, seed=None, params=None, extra=None,
               capacity=None, strict=False):
    """
    Standardized logs for solvers with sanity checks.

    Args:
      message (str): human readable status.
      runtime (float): elapsed seconds.
      final_value (float|None): objective value returned by solver (before zero-weight items added).
      final_weight (int|float|None): total weight of the returned solution.
      solution_size (int|None): number of items in returned solution.
      seed (int|None): RNG seed used.
      params (dict|None): solver parameters for reproducibility.
      extra (dict|None): any extra fields (e.g. convergence_history).
      capacity (int|float|None): instance capacity; when provided we will sanity-check final_weight <= capacity.
      strict (bool): if True, raise AssertionError on sanity violation (useful for tests). Default False.

    Returns:
      dict: structured log containing inputs above plus 'sanity_warnings' (list) and 'infeasible' (bool).
    """
    logs = {
        "message": str(message),
        "runtime": float(runtime) if runtime is not None else None,
        "final_value": None if final_value is None else float(final_value),
        "final_weight": None if final_weight is None else float(final_weight),
        "solution_size": None if solution_size is None else int(solution_size),
        "seed": seed,
        "params": params or {},
        "extra": extra or {},
        "timestamp": time.time(),
        # fields we will populate below
        "sanity_warnings": [],
        "infeasible": False
    }

    # 1) final_value must be finite (not inf/-inf/nan)
    if logs["final_value"] is not None:
        if not math.isfinite(logs["final_value"]):
            msg = f"final_value is not finite ({logs['final_value']})"
            if strict:
                raise AssertionError(msg)
            # sanitize: set to 0.0 (safe neutral) and warn
            logs["sanity_warnings"].append(msg + " -> reset to 0.0")
            logs["final_value"] = 0.0

    # 2) final_weight vs capacity check
    if logs["final_weight"] is not None and capacity is not None:
        try:
            cap_val = float(capacity)
        except Exception:
            cap_val = None

        if cap_val is not None:
            if logs["final_weight"] > cap_val:
                msg = f"final_weight ({logs['final_weight']}) exceeds capacity ({cap_val})"
                if strict:
                    raise AssertionError(msg)
                # Mark infeasible, don't silently clip weight (better to preserve observed value)
                logs["sanity_warnings"].append(msg + " -> marked infeasible in logs")
                logs["infeasible"] = True

    # 3) additional basic sanity checks
    if logs["solution_size"] is not None and logs["solution_size"] < 0:
        msg = f"solution_size ({logs['solution_size']}) negative"
        if strict:
            raise AssertionError(msg)
        logs["sanity_warnings"].append(msg + " -> set to 0")
        logs["solution_size"] = max(0, logs["solution_size"])

    # 4) unify types for downstream consumers
    # keep final_value and final_weight as floats (or None)
    # keep solution_size as int (or None)

    return logs

def _make_base_logs(message, runtime, final_value=float('nan'), final_weight=None,
                    solution_size=0, seed=None, params=None, extra=None, capacity=None):
    """Minimal logs factory used if your module doesn't expose _base_logs (self-contained)."""
    return {
        "message": message,
        "runtime": float(runtime),
        "final_value": None if final_value is None else float(final_value) if (not (isinstance(final_value, float) and math.isnan(final_value))) else float('nan'),
        "final_weight": None if final_weight is None else float(final_weight),
        "solution_size": int(solution_size) if solution_size is not None else 0,
        "seed": seed,
        "params": params or {},
        "extra": extra or {},
        "capacity": capacity
    }

# ==============================================================================
# --- Amod's Solvers ---
# ==============================================================================

@accept_instance
def dynamic_programming_solver(items, capacity, timeout=None, seed=None, memory_guard=2e7):
    """
    1D dynamic programming knapsack solver with reconstruction and timeout handling.

    Returns (chosen_item_ids, total_value, logs)
    """
    start = time.perf_counter()
    n = len(items)

    # Preprocess: zero-weight positive-value items (always take)
    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero_items = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero_items.append({'id': it['id'], 'weight': w, 'value': v})

    n_nonzero = len(nonzero_items)

    # Memory guard check: warn if the full 2D would be large
    approx_cells = (n_nonzero + 1) * (capacity + 1)
    if approx_cells > memory_guard:
        logging.getLogger(__name__).warning(
            "DP large: approx cells=%d exceeds memory_guard=%d. Using 1D DP + choice matrix (may be large).",
            approx_cells, memory_guard
        )

    # Try to allocate 1D dp and choice matrix for reconstruction
    try:
        dp = [0.0] * (capacity + 1)
        choice = [bytearray(capacity + 1) for _ in range(n_nonzero)]
    except MemoryError:
        logging.getLogger(__name__).warning(
            "MemoryError creating choice matrix. Falling back to value-only DP (no reconstruction)."
        )
        dp = [0.0] * (capacity + 1)
        choice = None

    # DP forward pass
    for i, itm in enumerate(nonzero_items):
        w = itm['weight']
        v = itm['value']

        for c in range(capacity, w - 1, -1):
            # coarse timeout check
            if timeout is not None and (time.perf_counter() - start) > timeout:
                # Return best-so-far
                if choice is not None:
                    best_c = max(range(capacity + 1), key=lambda x: dp[x])
                    chosen_ids = []
                    cur_c = best_c
                    # reconstruct from items seen so far (i down to 0)
                    for j in range(i, -1, -1):
                        if cur_c >= 0 and choice[j][cur_c]:
                            chosen_ids.append(nonzero_items[j]['id'])
                            cur_c -= nonzero_items[j]['weight']
                    chosen_ids = zero_weight_ids + list(reversed(chosen_ids))
                    total_value = dp[best_c] + zero_weight_value
                else:
                    best_c = max(range(capacity + 1), key=lambda x: dp[x])
                    chosen_ids = zero_weight_ids.copy()
                    total_value = dp[best_c] + zero_weight_value

                logs = _base_logs(
                    "timed out; returning best-so-far (DP)",
                    time.perf_counter() - start,
                    final_value=total_value,
                    solution_size=len(chosen_ids),
                    seed=seed,
                    params={"method": "1D_DP_partial"}
                )
                return chosen_ids, float(total_value), logs

            new_val = v + dp[c - w]
            if new_val > dp[c]:
                dp[c] = new_val
                if choice is not None:
                    choice[i][c] = 1

    # Completed DP. Reconstruct optimal solution
    best_c = max(range(capacity + 1), key=lambda x: dp[x])
    total_value = dp[best_c] + zero_weight_value
    chosen_ids = []
    if choice is not None:
        cur_c = best_c
        for i in range(n_nonzero - 1, -1, -1):
            if cur_c >= 0 and choice[i][cur_c]:
                chosen_ids.append(nonzero_items[i]['id'])
                cur_c -= nonzero_items[i]['weight']
        chosen_ids = zero_weight_ids + list(reversed(chosen_ids))
    else:
        # no reconstruction info => only zero-weight items are returned
        chosen_ids = zero_weight_ids.copy()
        logging.getLogger(__name__).warning("DP: no reconstruction available; returning zero-weight items only.")

    logs = _base_logs(
        "DP finished",
        time.perf_counter() - start,
        final_value=total_value,
        solution_size=len(chosen_ids),
        seed=seed,
        params={"method": "1D_DP_full", "n_nonzero": n_nonzero}
    )
    return chosen_ids, float(total_value), logs

@accept_instance
def greedy_ratio_solver(items, capacity, timeout=None, seed=None, check_period=1000):
    """
    Greedy by value/weight ratio with deterministic tie-breaker (id).
    Returns a feasible best-so-far solution on timeout.
    """
    start = time.perf_counter()

    # Preprocess zero-weight positive-value items
    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})

    # Sort by ratio desc, tie-breaker id
    nonzero.sort(key=lambda x: (-(x['value'] / x['weight']), x['id']))

    chosen_ids = []
    total_weight = 0
    total_value = 0.0

    for idx, it in enumerate(nonzero):
        if timeout is not None and (idx % check_period == 0):
            if (time.perf_counter() - start) > timeout:
                best_solution_ids = zero_weight_ids + chosen_ids
                logs = _base_logs(
                    "timed out; returning best-so-far (greedy_ratio)",
                    time.perf_counter() - start,
                    final_value=total_value + zero_weight_value,
                    final_weight=total_weight,
                    solution_size=len(best_solution_ids),
                    seed=seed,
                    params={"method": "greedy_ratio", "check_period": check_period}
                )
                return best_solution_ids, float(total_value + zero_weight_value), logs

        if total_weight + it['weight'] <= capacity:
            chosen_ids.append(it['id'])
            total_weight += it['weight']
            total_value += it['value']

    final_ids = zero_weight_ids + chosen_ids
    logs = _base_logs(
        "greedy_ratio finished",
        time.perf_counter() - start,
        final_value=total_value + zero_weight_value,
        final_weight=total_weight,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "greedy_ratio"}
    )
    return final_ids, float(total_value + zero_weight_value), logs

@accept_instance
def genetic_algorithm_solver(items, capacity, population_size=50, generations=200,
                             crossover_rate=0.8, mutation_rate=0.02, elitism=1,
                             tournament_k=3, timeout=None, seed=None, check_period=5):
    """
    GA for knapsack with repair to guarantee feasibility and proper logging.
    Includes early-detection + fallback to avoid all-zero fitness runs.
    """
    start = time.perf_counter()
    rnd = random.Random(seed)

    # Preprocess zero-weight items
    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})
    n = len(nonzero)
    if n == 0:
        logs = _base_logs("no non-zero-weight items", time.perf_counter() - start, final_value=zero_weight_value, solution_size=len(zero_weight_ids), seed=seed, capacity=capacity)
        return zero_weight_ids, float(zero_weight_value), logs

    # Helper: individual as bitlist of length n
    def random_individual():
        return [rnd.choice([0, 1]) for _ in range(n)]

    def evaluate(ind):
        total_w = 0
        total_v = 0.0
        for bit, it in zip(ind, nonzero):
            if bit:
                total_w += it['weight']
                total_v += it['value']
        if total_w > capacity:
            return 0.0, total_w  # infeasible -> score 0, but return weight
        return float(total_v), total_w

    # repair: if infeasible, remove included items with smallest value/weight until feasible
    def repair(ind):
        ind = ind[:]  # copy
        total_w = 0
        for bit, it in zip(ind, nonzero):
            if bit:
                total_w += it['weight']
        if total_w <= capacity:
            return ind
        # compute included indices sorted by ascending value/weight (worst contributions removed first)
        included = [i for i, b in enumerate(ind) if b]
        included.sort(key=lambda i: (nonzero[i]['value'] / nonzero[i]['weight'], nonzero[i]['id']))
        for idx in included:
            if total_w <= capacity:
                break
            if ind[idx] == 1:
                ind[idx] = 0
                total_w -= nonzero[idx]['weight']
        return ind

    # initialize population (repair infeasible individuals)
    population = [repair(random_individual()) for _ in range(population_size)]
    fitness = [evaluate(ind)[0] for ind in population]

    # If initial population all-zero fitness -> try repair once, else fallback to greedy immediately
    if max(fitness) == 0.0:
        try:
            population = [repair(ind) for ind in population]
            fitness = [evaluate(ind)[0] for ind in population]
        except Exception:
            # fallback
            final_ids, fallback_val, fallback_logs = greedy_ratio_solver(items=items, capacity=capacity, seed=seed)
            logs = _base_logs(
                "GA early-detect: all-zero fitness on init -> fallback greedy",
                time.perf_counter() - start,
                final_value=float(fallback_val + zero_weight_value),
                solution_size=len(final_ids),
                seed=seed,
                params={"reason": "all-zero-fitness-on-init"}
            )
            return final_ids, float(fallback_val + zero_weight_value), logs

    # track best feasible individual explicitly
    best_feasible_val = -1.0
    best_feasible_idx = None
    for i, ind in enumerate(population):
        val, wt = evaluate(ind)
        if wt <= capacity and val > best_feasible_val:
            best_feasible_val = val
            best_feasible_idx = i

    convergence_history = []
    params = {
        "population_size": population_size,
        "generations": generations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "elitism": elitism,
        "tournament_k": tournament_k
    }

    def tournament_select():
        best = None
        best_fit = -1
        for _ in range(tournament_k):
            i = rnd.randrange(population_size)
            if fitness[i] > best_fit:
                best = i
                best_fit = fitness[i]
        return population[best][:]

    def crossover(a, b):
        if rnd.random() > crossover_rate:
            return a[:], b[:]
        pt = rnd.randrange(1, n)
        child1 = a[:pt] + b[pt:]
        child2 = b[:pt] + a[pt:]
        return child1, child2

    def mutate(ind):
        for i in range(n):
            if rnd.random() < mutation_rate:
                ind[i] = 1 - ind[i]

    # Zero-progress detector
    zero_gen_counter = 0
    ZERO_GEN_LIMIT = 10

    # Main GA loop
    for gen in range(1, generations + 1):
        # timeout (coarse-grained per generation)
        if timeout is not None and (time.perf_counter() - start) > timeout:
            # ensure we return a feasible solution if one exists
            if best_feasible_idx is not None:
                best_ind = population[best_feasible_idx]
                best_val, best_w = evaluate(best_ind)
                chosen_ids = [nonzero[i]['id'] for i, b in enumerate(best_ind) if b]
                final_ids = zero_weight_ids + chosen_ids
                logs = _base_logs(
                    "timed out; returning best-so-far (GA)",
                    time.perf_counter() - start,
                    final_value=best_val + zero_weight_value,
                    final_weight=best_w,
                    solution_size=len(final_ids),
                    seed=seed,
                    params=params,
                    extra={"generations_ran": gen - 1, "convergence_history": convergence_history},
                    capacity=capacity
                )
                return final_ids, float(best_val + zero_weight_value), logs
            else:
                # fallback to greedy_ratio if GA produced no feasible solution
                final_ids, fallback_val, fallback_logs = greedy_ratio_solver(items=items, capacity=capacity, seed=seed)
                logs = _base_logs(
                    "timed out; GA produced no feasible solutions -> returned greedy fallback",
                    time.perf_counter() - start,
                    final_value=fallback_val,
                    solution_size=len(final_ids),
                    seed=seed,
                    params=params,
                    extra={"generations_ran": gen - 1, "convergence_history": convergence_history},
                    capacity=capacity
                )
                return final_ids, float(fallback_val + zero_weight_value), logs

        # create new population with elitism
        new_pop = []
        sorted_idx = sorted(range(population_size), key=lambda i: fitness[i], reverse=True)
        for e in range(min(elitism, population_size)):
            new_pop.append(population[sorted_idx[e]][:])

        while len(new_pop) < population_size:
            p1 = tournament_select()
            p2 = tournament_select()
            c1, c2 = crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            # repair children before adding
            c1 = repair(c1)
            new_pop.append(c1)
            if len(new_pop) < population_size:
                c2 = repair(c2)
                new_pop.append(c2)

        population = new_pop
        fitness = [evaluate(ind)[0] for ind in population]

        # update best feasible
        for i, ind in enumerate(population):
            val, wt = evaluate(ind)
            if wt <= capacity and val > best_feasible_val:
                best_feasible_val = val
                best_feasible_idx = i

        best_val_gen = max(fitness)
        convergence_history.append(float(best_val_gen + zero_weight_value))

        # zero-progress detection and fallback
        if best_val_gen == 0.0:
            zero_gen_counter += 1
        else:
            zero_gen_counter = 0

        if zero_gen_counter >= ZERO_GEN_LIMIT:
            # try repair once more across population
            try:
                population = [repair(ind) for ind in population]
                fitness = [evaluate(ind)[0] for ind in population]
                zero_gen_counter = 0
                # if still zero after repair, fallback
                if max(fitness) == 0.0:
                    raise RuntimeError("repair did not resolve zero fitness")
            except Exception:
                final_ids, fallback_val, fallback_logs = greedy_ratio_solver(items=items, capacity=capacity, seed=seed)
                logs = _base_logs(
                    "GA no-progress: fallback to greedy after repeated zero fitness generations",
                    time.perf_counter() - start,
                    final_value=float(fallback_val + zero_weight_value),
                    solution_size=len(final_ids),
                    seed=seed,
                    params={"generations_ran": gen, "reason": "no_progress_zero_fitness"},
                    capacity=capacity
                )
                return final_ids, float(fallback_val + zero_weight_value), logs

    # finished all generations: return best feasible if any, else greedy fallback
    if best_feasible_idx is not None:
        best_ind = population[best_feasible_idx]
        best_val, best_w = evaluate(best_ind)
        chosen_ids = [nonzero[i]['id'] for i, b in enumerate(best_ind) if b]
        final_ids = zero_weight_ids + chosen_ids
        logs = _base_logs(
            "GA finished",
            time.perf_counter() - start,
            final_value=best_val + zero_weight_value,
            final_weight=best_w,
            solution_size=len(final_ids),
            seed=seed,
            params=params,
            extra={"generations_ran": generations, "convergence_history": convergence_history},
            capacity=capacity
        )
        return final_ids, float(best_val + zero_weight_value), logs
    else:
        # fallback: no feasible GA solution found at all -> return greedy_ratio
        final_ids, fallback_val, fallback_logs = greedy_ratio_solver(items=items, capacity=capacity, seed=seed)
        logs = _base_logs(
            "GA finished but produced no feasible solution -> returned greedy fallback",
            time.perf_counter() - start,
            final_value=fallback_val,
            solution_size=len(final_ids),
            seed=seed,
            params=params,
            extra={"generations_ran": generations, "convergence_history": convergence_history},
            capacity=capacity
        )
        return final_ids, float(fallback_val + zero_weight_value), logs

# ==============================================================================
# --- Kartik's Solvers ---
# ==============================================================================

@accept_instance
def branch_and_bound_solver(items, capacity, timeout=None, seed=None, check_period=1000):
    """
    Branch-and-Bound solver for 0/1 knapsack with safety guards:
      - handles zero-weight (positive-value) items separately
      - prevents returning -inf as a final_value when no feasible leaf was found
      - respects timeout (coarse-grained)
      - returns logs via _base_logs (assumes that helper exists in my_solvers.py)
    Replace the previous branch_and_bound_solver with this function.
    """
    start = time.perf_counter()
    rnd = random.Random(seed)

    # Separate out zero-weight positive-value items (they always belong to solution)
    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get("weight", 0))
        v = float(it.get("value", 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it["id"])
                zero_weight_value += v
        else:
            nonzero.append({"id": it["id"], "weight": w, "value": v})
    n = len(nonzero)
    if n == 0:
        # nothing with positive weight -> trivial solution
        final_ids = zero_weight_ids[:]
        logs = _base_logs(
            "no non-zero-weight items",
            time.perf_counter() - start,
            final_value=float(zero_weight_value),
            solution_size=len(final_ids),
            seed=seed,
            params={"method": "branch_and_bound", "nodes_explored": 0}
        )
        return final_ids, float(zero_weight_value), logs

    # Precompute value/weight ratio ordering (use for bound)
    order = sorted(range(n), key=lambda i: nonzero[i]["value"] / nonzero[i]["weight"], reverse=True)
    inv_order = {ordered_idx: idx for idx, ordered_idx in enumerate(order)}
    w_ordered = [nonzero[i]["weight"] for i in order]
    v_ordered = [nonzero[i]["value"] for i in order]

    # Helper: fractional upper bound (classic knapsack fractional relaxation)
    def fractional_bound(start_idx, remaining_cap, acc_value):
        val = acc_value
        cap = remaining_cap
        i = start_idx
        while i < n and cap > 0:
            w = w_ordered[i]
            v = v_ordered[i]
            if w <= cap:
                val += v
                cap -= w
            else:
                # fraction
                val += v * (cap / w)
                cap = 0
            i += 1
        return val

    # B&B state: stack of (index, current_value, current_weight, chosen_bits_list)
    # index refers to position in ordered lists
    stack = [(0, 0.0, 0, [0] * n)]
    best_value = float("-inf")
    best_choice = [0] * n
    nodes = 0
    last_check = 0

    # Iterative DFS with bounding
    while stack:
        # Timeout coarse check
        if timeout is not None and (time.perf_counter() - start) > timeout:
            # Safe-guard: don't return -inf, use 0.0 if no feasible solution known
            safe_best_value = 0.0 if best_value == float("-inf") else best_value
            chosen_ids = [nonzero[order[i]]["id"] for i, b in enumerate(best_choice) if b]
            final_ids = zero_weight_ids + chosen_ids
            logs = _base_logs(
                "timed out; returning best-so-far (B&B)",
                time.perf_counter() - start,
                final_value=float(safe_best_value + zero_weight_value),
                solution_size=len(final_ids),
                seed=seed,
                params={"method": "branch_and_bound", "nodes_explored": nodes}
            )
            return final_ids, float(safe_best_value + zero_weight_value), logs

        idx, cur_val, cur_wt, bits = stack.pop()
        nodes += 1

        # Periodic sanity check to avoid pathological loops
        last_check += 1
        if last_check >= check_period:
            last_check = 0
            # sanity: if time exceeded, break (handled above)
            pass

        # If weight already over capacity -> prune
        if cur_wt > capacity:
            continue

        # If at leaf node, update best (feasible)
        if idx == n:
            if cur_val > best_value:
                best_value = cur_val
                best_choice = bits.copy()
            continue

        # Bound: max possible value from this node
        bound = fractional_bound(idx, capacity - cur_wt, cur_val)
        # If bound <= best_value, prune
        if bound <= best_value:
            continue

        # Expand children: try include and exclude for the next item (ordered indexing)
        # include
        w_inc = cur_wt + w_ordered[idx]
        v_inc = cur_val + v_ordered[idx]
        if w_inc <= capacity:
            bits_inc = bits[:]
            bits_inc[idx] = 1
            stack.append((idx + 1, v_inc, w_inc, bits_inc))
        # exclude
        bits_exc = bits[:]
        bits_exc[idx] = 0
        stack.append((idx + 1, cur_val, cur_wt, bits_exc))

    # Finished traversal. If best_value stayed -inf, we found no feasible leaf (should be rare).
    safe_best_value = 0.0 if best_value == float("-inf") else best_value
    chosen_ids = [nonzero[order[i]]["id"] for i, b in enumerate(best_choice) if b]
    final_ids = zero_weight_ids + chosen_ids

    logs = _base_logs(
        "branch_and_bound finished",
        time.perf_counter() - start,
        final_value=float(safe_best_value + zero_weight_value),
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "branch_and_bound", "nodes_explored": nodes}
    )
    return final_ids, float(safe_best_value + zero_weight_value), logs

@accept_instance
def greedy_value_solver(items, capacity, timeout=None, seed=None, check_period=1000):
    """
    Greedy by absolute value (descending). Returns feasible selection, best-so-far on timeout.
    """
    start = time.perf_counter()

    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})

    # sort by value desc, tie-breaker id
    nonzero.sort(key=lambda x: ( -x['value'], x['id']))

    chosen_ids = []
    total_weight = 0
    total_value = 0.0

    for idx, it in enumerate(nonzero):
        if timeout is not None and (idx % check_period == 0):
            if (time.perf_counter() - start) > timeout:
                final_ids = zero_weight_ids + chosen_ids
                return final_ids, float(total_value + zero_weight_value), _base_logs(
                    "timed out; returning best-so-far (greedy_value)",
                    time.perf_counter() - start,
                    final_value=total_value + zero_weight_value,
                    final_weight=total_weight,
                    solution_size=len(final_ids),
                    seed=seed,
                    params={"method": "greedy_value"}
                )

        if total_weight + it['weight'] <= capacity:
            chosen_ids.append(it['id'])
            total_weight += it['weight']
            total_value += it['value']

    final_ids = zero_weight_ids + chosen_ids
    return final_ids, float(total_value + zero_weight_value), _base_logs(
        "greedy_value finished",
        time.perf_counter() - start,
        final_value=total_value + zero_weight_value,
        final_weight=total_weight,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "greedy_value"}
    )

@accept_instance
def ant_colony_solver(items, capacity, ants=30, iterations=100, alpha=1.0, beta=2.0,
                      rho=0.1, q=1.0, timeout=None, seed=None):
    """
    Simple ACO for knapsack: pheromone for each item; ants build solutions by sampling ordering
    biased by pheromone^alpha * (value/weight)^beta and select until capacity.
    Returns best-so-far and convergence history.
    """
    start = time.perf_counter()
    rnd = random.Random(seed)

    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})
    n = len(nonzero)
    if n == 0:
        return zero_weight_ids, float(zero_weight_value), _base_logs("no nonzero items", time.perf_counter()-start, final_value=zero_weight_value, solution_size=len(zero_weight_ids), seed=seed)

    # Initialize pheromone
    pher = [1.0] * n
    heuristic = [it['value'] / it['weight'] for it in nonzero]

    best_value = 0.0
    best_solution = []
    convergence_history = []

    for iters in range(1, iterations + 1):
        if timeout is not None and (time.perf_counter() - start) > timeout:
            final_ids = zero_weight_ids + best_solution
            return final_ids, float(best_value + zero_weight_value), _base_logs(
                "timed out; returning best-so-far (ACO)",
                time.perf_counter() - start,
                final_value=best_value + zero_weight_value,
                solution_size=len(final_ids),
                seed=seed,
                params={"method": "ant_colony", "iterations_done": iters, "ants": ants},
                extra={"convergence_history": convergence_history}
            )

        all_solutions = []
        all_values = []
        for a in range(ants):
            # build probabilities
            desirabilities = [ (pher[i] ** alpha) * (heuristic[i] ** beta) for i in range(n) ]
            # create probability distribution for ordering: sample a permutation by weighted choice
            # we'll create a randomized ranking by sampling without replacement using weights
            remaining = list(range(n))
            order = []
            local_des = desirabilities[:]
            while remaining:
                total = sum(local_des[i] for i in remaining)
                if total <= 0:
                    # uniform tie-break
                    choice = rnd.choice(remaining)
                else:
                    r = rnd.random() * total
                    s = 0.0
                    chosen = remaining[0]
                    for idx_choice in remaining:
                        s += local_des[idx_choice]
                        if s >= r:
                            chosen = idx_choice
                            break
                    choice = chosen
                order.append(choice)
                remaining.remove(choice)

            # greedily add items in this order
            sol = []
            wsum = 0
            vsum = 0.0
            for idx_item in order:
                it = nonzero[idx_item]
                if wsum + it['weight'] <= capacity:
                    sol.append(idx_item)
                    wsum += it['weight']
                    vsum += it['value']
            all_solutions.append(sol)
            all_values.append(vsum)

        # pheromone evaporation
        pher = [ (1 - rho) * p for p in pher ]

        # deposit pheromone proportional to solution quality (use best ant of this iteration)
        if all_values:
            iter_best_idx = max(range(len(all_values)), key=lambda i: all_values[i])
            iter_best_val = all_values[iter_best_idx]
            iter_best_sol = all_solutions[iter_best_idx]
            if iter_best_val > best_value:
                best_value = iter_best_val
                best_solution = [nonzero[i]['id'] for i in iter_best_sol]

            # update pheromone
            for idx_item in iter_best_sol:
                pher[idx_item] += q * iter_best_val

        convergence_history.append(float(best_value + zero_weight_value))

    final_ids = zero_weight_ids + best_solution
    logs = _base_logs(
        "ACO finished",
        time.perf_counter() - start,
        final_value=best_value + zero_weight_value,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "ant_colony", "iterations": iterations, "ants": ants},
        extra={"convergence_history": convergence_history}
    )
    return final_ids, float(best_value + zero_weight_value), logs

# ==============================================================================
# --- Sudarshan's Solvers ---
# ==============================================================================

@accept_instance
def backtracking_solver(items, capacity, timeout=None, seed=None):
    """
    Backtracking solver using same fractional bound as B&B.
    Similar to branch_and_bound but implemented recursively for clarity.
    """
    start = time.perf_counter()

    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})

    nonzero.sort(key=lambda x: (-(x['value'] / x['weight']), x['id']))
    n = len(nonzero)
    if n == 0:
        return zero_weight_ids, float(zero_weight_value), _base_logs("no nonzero items", time.perf_counter()-start, final_value=zero_weight_value, solution_size=len(zero_weight_ids), seed=seed)

    def fractional_bound(i, cur_w, cur_v):
        if cur_w >= capacity:
            return cur_v
        bound = cur_v
        w_rem = capacity - cur_w
        for j in range(i, n):
            it = nonzero[j]
            if it['weight'] <= w_rem:
                w_rem -= it['weight']
                bound += it['value']
            else:
                bound += it['value'] * (w_rem / it['weight'])
                break
        return bound

    best_value = float('-inf')
    best_choice = [0] * n
    nodes = 0

    def dfs(i, cur_w, cur_v, choice):
        nonlocal best_value, best_choice, nodes
        nodes += 1
        if timeout is not None and (time.perf_counter() - start) > timeout:
            return  # just unwind; caller will check best-so-far

        if cur_w > capacity:
            return
        if i == n:
            if cur_v > best_value:
                best_value = cur_v
                best_choice = choice[:]
            return

        bound = fractional_bound(i, cur_w, cur_v)
        if bound <= best_value:
            return

        # choose item
        choice[i] = 1
        dfs(i + 1, cur_w + nonzero[i]['weight'], cur_v + nonzero[i]['value'], choice)
        # unchoose
        choice[i] = 0
        dfs(i + 1, cur_w, cur_v, choice)

    dfs(0, 0, 0.0, [0]*n)

    # Check timeout possibility (we don't detect exact time inside recursion beyond the checks that return early)
    chosen_ids = [nonzero[i]['id'] for i, b in enumerate(best_choice) if b]
    final_ids = zero_weight_ids + chosen_ids
    logs = _base_logs(
        "backtracking finished",
        time.perf_counter() - start,
        final_value=best_value + zero_weight_value if best_value != float('-inf') else zero_weight_value,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "backtracking", "nodes_explored": nodes}
    )
    return final_ids, float((best_value + zero_weight_value) if best_value != float('-inf') else zero_weight_value), logs

@accept_instance
def greedy_weight_solver(items, capacity, timeout=None, seed=None, check_period=1000):
    """
    Greedy by smallest weight first. Useful baseline.
    """
    start = time.perf_counter()

    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w = int(it.get('weight', 0))
        v = float(it.get('value', 0.0))
        if w <= 0:
            if v > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v
        else:
            nonzero.append({'id': it['id'], 'weight': w, 'value': v})

    # sort by weight asc, tie-breaker id
    nonzero.sort(key=lambda x: (x['weight'], x['id']))

    chosen_ids = []
    total_weight = 0
    total_value = 0.0

    for idx, it in enumerate(nonzero):
        if timeout is not None and (idx % check_period == 0):
            if (time.perf_counter() - start) > timeout:
                final_ids = zero_weight_ids + chosen_ids
                return final_ids, float(total_value + zero_weight_value), _base_logs(
                    "timed out; returning best-so-far (greedy_weight)",
                    time.perf_counter() - start,
                    final_value=total_value + zero_weight_value,
                    final_weight=total_weight,
                    solution_size=len(final_ids),
                    seed=seed,
                    params={"method": "greedy_weight"}
                )

        if total_weight + it['weight'] <= capacity:
            chosen_ids.append(it['id'])
            total_weight += it['weight']
            total_value += it['value']

    final_ids = zero_weight_ids + chosen_ids
    return final_ids, float(total_value + zero_weight_value), _base_logs(
        "greedy_weight finished",
        time.perf_counter() - start,
        final_value=total_value + zero_weight_value,
        final_weight=total_weight,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "greedy_weight"}
    )

@accept_instance
def particle_swarm_solver(items, capacity, particles=30, iterations=200, w=0.7, c1=1.4, c2=1.4, timeout=None, seed=None):
    """
    Binary-PSO: particle positions in R, mapped to 0/1 via sigmoid; velocity in R.
    Returns best-so-far solution and convergence history.
    """
    start = time.perf_counter()
    rnd = random.Random(seed)

    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    for it in items:
        w_i = int(it.get('weight', 0))
        v_i = float(it.get('value', 0.0))
        if w_i <= 0:
            if v_i > 0:
                zero_weight_ids.append(it['id'])
                zero_weight_value += v_i
        else:
            nonzero.append({'id': it['id'], 'weight': w_i, 'value': v_i})
    n = len(nonzero)
    if n == 0:
        return zero_weight_ids, float(zero_weight_value), _base_logs("no nonzero items", time.perf_counter()-start, final_value=zero_weight_value, solution_size=len(zero_weight_ids), seed=seed)

    # initialize particles
    # positions and velocities are lists of floats of length n
    positions = [[rnd.random() for _ in range(n)] for _ in range(particles)]
    velocities = [[rnd.uniform(-1, 1) for _ in range(n)] for _ in range(particles)]

    # personal bests
    pbest_pos = [pos[:] for pos in positions]
    pbest_val = []
    pbest_weight = []

    def decode(pos):
        # map pos floats to binary via sigmoid threshold 0.5
        sol_bits = [1 if (1 / (1 + math.exp(-p))) > 0.5 else 0 for p in pos]
        # ensure feasibility: greedy remove lowest ratio included items until feasible
        total_w = 0
        total_v = 0.0
        included_idx = []
        for i, bit in enumerate(sol_bits):
            if bit:
                total_w += nonzero[i]['weight']
                total_v += nonzero[i]['value']
                included_idx.append(i)
        if total_w <= capacity:
            return sol_bits, total_v, total_w
        # remove items with lowest value/weight first
        included_idx.sort(key=lambda i: (nonzero[i]['value'] / nonzero[i]['weight'], nonzero[i]['id']))  # ascending ratio removal
        for idx_rm in included_idx:
            # remove until feasible
            if sol_bits[idx_rm] == 1:
                sol_bits[idx_rm] = 0
                total_w -= nonzero[idx_rm]['weight']
                total_v -= nonzero[idx_rm]['value']
            if total_w <= capacity:
                break
        return sol_bits, total_v, total_w

    # evaluate initial personal bests
    for pos in positions:
        _, val, wt = decode(pos)
        pbest_val.append(val)
        pbest_weight.append(wt)

    # global best
    gbest_idx = max(range(particles), key=lambda i: pbest_val[i])
    gbest_pos = pbest_pos[gbest_idx][:]
    gbest_val = pbest_val[gbest_idx]
    gbest_weight = pbest_weight[gbest_idx]

    convergence_history = [float(gbest_val + zero_weight_value)]

    for iters in range(1, iterations + 1):
        if timeout is not None and (time.perf_counter() - start) > timeout:
            # return best-so-far
            chosen_ids = [nonzero[i]['id'] for i, b in enumerate(decode(gbest_pos)[0]) if b]
            final_ids = zero_weight_ids + chosen_ids
            return final_ids, float(gbest_val + zero_weight_value), _base_logs(
                "timed out; returning best-so-far (PSO)",
                time.perf_counter() - start,
                final_value=gbest_val + zero_weight_value,
                final_weight=gbest_weight,
                solution_size=len(final_ids),
                seed=seed,
                params={"method": "particle_swarm", "particles": particles, "iterations_done": iters},
                extra={"convergence_history": convergence_history}
            )

        for p in range(particles):
            # update velocity and position
            for d in range(n):
                r1 = rnd.random()
                r2 = rnd.random()
                velocities[p][d] = w * velocities[p][d] + c1 * r1 * (pbest_pos[p][d] - positions[p][d]) + c2 * r2 * (gbest_pos[d] - positions[p][d])
                positions[p][d] += velocities[p][d]
            # evaluate
            _, val, wt = decode(positions[p])
            if val > pbest_val[p]:
                pbest_val[p] = val
                pbest_pos[p] = positions[p][:]
            # update global best
            if pbest_val[p] > gbest_val:
                gbest_val = pbest_val[p]
                gbest_pos = pbest_pos[p][:]
                gbest_weight = decode(gbest_pos)[2]

        convergence_history.append(float(gbest_val + zero_weight_value))

    # finished
    chosen_ids = [nonzero[i]['id'] for i, b in enumerate(decode(gbest_pos)[0]) if b]
    final_ids = zero_weight_ids + chosen_ids
    logs = _base_logs(
        "PSO finished",
        time.perf_counter() - start,
        final_value=gbest_val + zero_weight_value,
        final_weight=gbest_weight,
        solution_size=len(final_ids),
        seed=seed,
        params={"method": "particle_swarm", "particles": particles, "iterations": iterations},
        extra={"convergence_history": convergence_history}
    )
    return final_ids, float(gbest_val + zero_weight_value), logs

@accept_instance
def quantum_annealing_solver(items, capacity,
                             timeout=None, seed=None,
                             num_reads=500, penalty_multiplier=1.5, sampler_name=None,
                             check_period=1000):
    """
    Quantum annealing (simulated annealer) solver — no greedy fallback.
    Returns: (selected_ids, value_float, logs_dict)
      - On success returns feasible solution.
      - On failure returns ([], float('nan'), logs) where logs['message'] explains the cause.
    Accepts either decorated call (instance=...) if you use accept_instance decorator,
    or direct call quantum_annealing_solver(items, capacity, ...).
    """
    start = time.perf_counter()
    rnd = random.Random(seed)

    # local alias for logs factory — prefer module _base_logs if present
    try:
        _base_logs = globals().get("_base_logs", None) or globals().get("base_logs", None)
        if not callable(_base_logs):
            raise NameError
    except Exception:
        _base_logs = _make_base_logs

    # --- preprocess items ---
    zero_weight_ids = []
    zero_weight_value = 0.0
    nonzero = []
    try:
        for it in items:
            w = int(it.get('weight', 0))
            v = float(it.get('value', 0.0))
            if w <= 0:
                if v > 0:
                    zero_weight_ids.append(it['id'])
                    zero_weight_value += v
            else:
                nonzero.append({'id': it['id'], 'weight': w, 'value': v})
    except Exception as e:
        logs = _base_logs("invalid items structure", time.perf_counter() - start,
                          final_value=float('nan'), solution_size=0, seed=seed,
                          params={"exception": str(e)})
        return [], float('nan'), logs

    n = len(nonzero)
    if n == 0:
        logs = _base_logs("no non-zero-weight items", time.perf_counter() - start,
                          final_value=zero_weight_value, solution_size=len(zero_weight_ids),
                          seed=seed, capacity=capacity)
        return zero_weight_ids, float(zero_weight_value), logs

    if capacity <= 0:
        logs = _base_logs("capacity <= 0: only zero-weight items", time.perf_counter() - start,
                          final_value=zero_weight_value, solution_size=len(zero_weight_ids),
                          seed=seed, capacity=capacity)
        return zero_weight_ids, float(zero_weight_value), logs

    # Build BQM (require dimod)
    try:
        import dimod
    except Exception as e:
        logs = _base_logs("quantum_annealing: dimod not installed",
                          time.perf_counter() - start,
                          final_value=float('nan'),
                          solution_size=0,
                          seed=seed,
                          params={"reason": "dimod_missing", "exception": str(e)})
        return [], float('nan'), logs

    weights = [it['weight'] for it in nonzero]
    values = [it['value'] for it in nonzero]
    ids = [it['id'] for it in nonzero]

    # Penalty scaling
    max_v = max(values) if values else 1.0
    A = float(penalty_multiplier) * float(max_v)

    # Build linear/quadratic terms (careful: O(n^2) memory for quadratic)
    linear = {}
    for i in range(n):
        w_i = float(weights[i])
        v_i = float(values[i])
        linear[i] = (-v_i) + A * (w_i * w_i) - 2.0 * A * float(capacity) * w_i

    quadratic = {}
    # building full quadratic is O(n^2); large n may be heavy
    for i in range(n):
        wi = float(weights[i])
        for j in range(i + 1, n):
            q = 2.0 * A * wi * float(weights[j])
            if q != 0.0:
                quadratic[(i, j)] = q

    offset = A * (float(capacity) ** 2)
    try:
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset=offset, vartype=dimod.BINARY)
    except Exception as e:
        logs = _base_logs("failed to build BQM", time.perf_counter() - start,
                          final_value=float('nan'),
                          solution_size=0,
                          seed=seed,
                          params={"exception": str(e), "n": n})
        return [], float('nan'), logs

    # Choose sampler: prefer neal, else dimod.SimulatedAnnealingSampler, else fail (no fallback)
    sampler_used = None
    sampler = None
    try:
        import neal
        sampler = neal.SimulatedAnnealingSampler()
        sampler_used = "neal.SimulatedAnnealingSampler"
    except Exception:
        try:
            sampler = dimod.SimulatedAnnealingSampler()
            sampler_used = "dimod.SimulatedAnnealingSampler"
        except Exception as e:
            logs = _base_logs("quantum_annealing: no sampler available",
                              time.perf_counter() - start,
                              final_value=float('nan'),
                              solution_size=0,
                              seed=seed,
                              params={"reason": "no_sampler_available", "exception": str(e)})
            return [], float('nan'), logs

    # Sampling parameters
    sample_kwargs = {"num_reads": int(num_reads)}
    if seed is not None:
        # different samplers expect different param names; supply both common ones
        sample_kwargs["seed"] = int(seed)
        sample_kwargs.setdefault("random_seed", int(seed))

    # Attempt sampling; on failure return explicit failure (no greedy fallback)
    try:
        response = sampler.sample(bqm, **sample_kwargs)
    except Exception as e:
        logs = _base_logs("quantum_annealing: sampler.sample failed",
                          time.perf_counter() - start,
                          final_value=float('nan'),
                          solution_size=0,
                          seed=seed,
                          params={"sampler_exception": str(e), "sampler_used": sampler_used})
        return [], float('nan'), logs

    # parse samples robustly: accept presence/absence of 'num_occ'
    best_sample = None
    best_energy = float("inf")
    feasible_found = False
    samples_checked = 0

    # iterate response entries robustly: support dict-like rows, tuple/list rows, or older APIs
    try:
        iterator = response.data(['sample', 'energy'])
    except Exception:
        # fallback: some samplers return plain iterable of (sample, energy, ...) tuples
        iterator = response

    for row in iterator:
        # normalize to (sample_dict, energy, num_occ)
        sample = None
        energy = None
        num_occ = 1
        try:
            # dict-like row (has .get)
            if hasattr(row, "get"):
                sample = row.get("sample", row.get("assignment", None) or row.get(0, None))
                energy = row.get("energy", row.get(1, None))
                # try a few candidate names for occurrence
                num_occ = int(row.get("num_occ", row.get("num_occurrence", row.get("num_occurrences", 1))))
            else:
                # tuple/list style: try common shapes
                # (sample, energy), (sample, energy, num_occ), etc.
                if isinstance(row, (tuple, list)):
                    if len(row) >= 2:
                        sample = row[0]
                        energy = row[1]
                    if len(row) >= 3:
                        try:
                            num_occ = int(row[2])
                        except Exception:
                            num_occ = 1
                else:
                    # last resort: try attribute access
                    sample = getattr(row, "sample", None)
                    energy = getattr(row, "energy", None)
                    num_occ = int(getattr(row, "num_occ", 1))
        except Exception:
            # If parsing row failed, skip it
            continue

        if sample is None or energy is None:
            # skip malformed rows
            continue

        try:
            num_occ = int(num_occ)
        except Exception:
            num_occ = 1

        samples_checked += max(1, num_occ)

        # compute weight & value of this sample
        total_w = 0
        total_v = 0.0
        try:
            for i in range(n):
                # sample may be dict mapping integer indices to 0/1 or mapping "0"-> etc.
                bit = 0
                if hasattr(sample, "get"):
                    bit = int(sample.get(i, sample.get(str(i), 0)))
                else:
                    # sample might be list/tuple or dimod.SampleView (supports __getitem__)
                    try:
                        bit = int(sample[i])
                    except Exception:
                        bit = int(sample.get(str(i), 0)) if hasattr(sample, "get") else 0
                if bit:
                    total_w += weights[i]
                    total_v += values[i]
        except Exception:
            # if sample indexing fails for some reason, skip this row
            continue

        if total_w <= capacity:
            feasible_found = True
            if energy < best_energy:
                best_energy = energy
                best_sample = (sample, energy)
        else:
            # still track best energy in case no feasible sample exists
            if best_sample is None or energy < best_energy:
                best_energy = energy
                best_sample = (sample, energy)

    if best_sample is None:
        logs = _base_logs("quantum_annealing: no valid samples returned",
                          time.perf_counter() - start,
                          final_value=float('nan'),
                          solution_size=0,
                          seed=seed,
                          params={"sampler_used": sampler_name or sampler_used})
        return [], float('nan'), logs

    sample, energy = best_sample

    # convert sample bits to chosen ids
    sel_bits = []
    try:
        for i in range(n):
            if hasattr(sample, "get"):
                sel_bits.append(int(sample.get(i, sample.get(str(i), 0))))
            else:
                try:
                    sel_bits.append(int(sample[i]))
                except Exception:
                    sel_bits.append(0)
    except Exception:
        # parsing problem -> fail
        logs = _base_logs("quantum_annealing: error parsing best sample bits",
                          time.perf_counter() - start,
                          final_value=float('nan'),
                          solution_size=0,
                          seed=seed,
                          params={"sampler_used": sampler_name or sampler_used})
        return [], float('nan'), logs

    total_w = sum(weights[i] for i, b in enumerate(sel_bits) if b)
    total_v = sum(values[i] for i, b in enumerate(sel_bits) if b)

    # if infeasible, do NOT repair (no fallback); mark as failure
    if total_w > capacity:
        logs = _base_logs("quantum_annealing: best sample infeasible (no repair performed)",
                          time.perf_counter() - start,
                          final_value=float('nan'),
                          final_weight=float(total_w),
                          solution_size=0,
                          seed=seed,
                          params={
                              "method": "quantum_annealing_simulated",
                              "sampler_used": sampler_name or sampler_used,
                              "num_reads": int(num_reads),
                              "penalty_multiplier": float(penalty_multiplier),
                              "A": float(A),
                              "feasible_sample_found": bool(feasible_found),
                              "samples_checked": int(samples_checked)
                          },
                          extra={"best_energy": float(energy)})
        return [], float('nan'), logs

    chosen_ids = [ids[i] for i, b in enumerate(sel_bits) if b]
    final_ids = zero_weight_ids + chosen_ids
    final_value = float(total_v + zero_weight_value)

    logs = _base_logs("quantum_annealing finished",
                      time.perf_counter() - start,
                      final_value=final_value,
                      final_weight=float(total_w),
                      solution_size=len(final_ids),
                      seed=seed,
                      params={
                          "method": "quantum_annealing_simulated",
                          "sampler_used": sampler_name or sampler_used,
                          "num_reads": int(num_reads),
                          "penalty_multiplier": float(penalty_multiplier),
                          "A": float(A),
                          "feasible_sample_found": bool(feasible_found),
                          "samples_checked": int(samples_checked)
                      },
                      extra={"best_energy": float(energy)},
                      capacity=capacity)

    return final_ids, final_value, logs

# ==============================================================================
# End of file
# ==============================================================================

