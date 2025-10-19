"""
Purpose:
- Generate reproducible 0/1 knapsack problem instances at multiple scales.
- Supports: seed-based reproducibility, distributions, positive/negative correlation,
  capacity ratio, batch generation and file output (JSON + CSV).

Usage:
- import functions or run as script to generate an example batch.

Notes:
- Reproducibility: each instance JSON stores the seed used in meta. 
  Regenerating the same instance with the same seed + parameters will produce identical items.
- Capacity_ratio controls hardness: smaller ratios generally make packing harder.
- (weight_dist / value_dist) let you simulate different real-world regimes 
    > uniform : uniform sampling
    > normal-ish : bell-curve around mid-range
    > heavy-tailed/zipf : few large, many small
- Correlation:
    > positive correlation: value proportional to weight with gaussian noise
    > negative correlation: value roughly inverse to weight with noise
"""

import os
import json
import csv
import random
from typing import List, Tuple, Optional, Dict
import math

# --- Utilities / RNG ---
def _get_rng(seed: Optional[int]):
    """Return (seed_used, random.Random instance)."""
    if seed is None:
        seed = random.randrange(0, 2**32)
    rng = random.Random(seed)
    return seed, rng

# --- Value/weight sampling functions ---
def _sample_uniform(rng: random.Random, low: int, high: int) -> int:
    return rng.randint(low, high)

def _sample_normal_int(rng: random.Random, mean: float, std: float, low: int, high: int) -> int:
    # draw until inside bounds to avoid extremes
    for _ in range(10):
        val = int(round(rng.gauss(mean, std)))
        if low <= val <= high:
            return val
    # fallback clamp
    return max(low, min(high, int(round(mean))))

def _sample_zipf_int(rng: random.Random, a: float, low: int, high: int) -> int:
    # simple discrete approximate zipf by sampling continuous and mapping
    # keep a small a (e.g., 1.2 - 2.0) for heavy-tail
    x = rng.random()
    # inverse transform for Pareto-like tail then clamp
    pareto = int(low + ( (1.0 - x) ** (-1.0/(a-1.0)) ) )
    return max(low, min(high, pareto))

# --- Instance generator ---
def generate_instance(
    n_items: int,
    weight_range: Tuple[int,int] = (1,100),
    value_range: Tuple[int,int] = (1,100),
    capacity_ratio: float = 0.5,
    correlation: Optional[str] = None,   # None | 'positive' | 'negative'
    weight_dist: str = "uniform",        # 'uniform' | 'normal' | 'zipf'
    value_dist: str = "uniform",         # same options
    seed: Optional[int] = None
) -> Dict:
    """
    Returns a dict:
    {
      "meta": { ... seed, params ... },
      "capacity": int,
      "items": [{"id": 0, "weight": w, "value": v}, ...]
    }
    Reproducible: instance['meta']['seed'] holds the RNG seed used.
    """
    # prepare RNG
    seed_used, rng = _get_rng(seed)

    wlow, whigh = weight_range
    vlow, vhigh = value_range

    # sampling helpers per distribution
    def sample_weight():
        if weight_dist == "uniform":
            return _sample_uniform(rng, wlow, whigh)
        elif weight_dist == "normal":
            mean = (wlow + whigh) / 2.0
            std = max(1.0, (whigh - wlow) / 6.0)
            return _sample_normal_int(rng, mean, std, wlow, whigh)
        elif weight_dist == "zipf":
            return _sample_zipf_int(rng, a=1.8, low=wlow, high=whigh)
        else:
            raise ValueError("Unknown weight_dist: " + str(weight_dist))

    def sample_value_from_weight(w):
        # If correlation not set, draw from value_dist; otherwise derive from weight + noise
        if correlation is None:
            if value_dist == "uniform":
                return _sample_uniform(rng, vlow, vhigh)
            elif value_dist == "normal":
                mean = (vlow + vhigh) / 2.0
                std = max(1.0, (vhigh - vlow) / 6.0)
                return _sample_normal_int(rng, mean, std, vlow, vhigh)
            elif value_dist == "zipf":
                return _sample_zipf_int(rng, a=1.8, low=vlow, high=vhigh)
            else:
                raise ValueError("Unknown value_dist: " + str(value_dist))
        else:
            wnorm = (w - wlow) / max(1, (whigh - wlow))
            if correlation == "positive":
                base = vlow + wnorm * (vhigh - vlow)
            else:  # 'negative'
                base = vlow + (1.0 - wnorm) * (vhigh - vlow)
            # add relative noise
            noise = rng.gauss(0, 0.08 * (vhigh - vlow))
            v = int(round(base + noise))
            return max(vlow, min(vhigh, v))

    items = []
    weights = []
    values = []

    for i in range(n_items):
        w = sample_weight()
        v = sample_value_from_weight(w)
        items.append({"id": i, "weight": int(w), "value": int(v)})
        weights.append(w); values.append(v)

    total_weight = sum(weights)
    # At least 1 capacity
    capacity = max(1, int(round(total_weight * capacity_ratio)))

    instance = {
        "meta": {
            "n_items": n_items,
            "weight_range": weight_range,
            "value_range": value_range,
            "capacity_ratio": capacity_ratio,
            "correlation": correlation,
            "weight_dist": weight_dist,
            "value_dist": value_dist,
            "seed": seed_used
        },
        "capacity": int(capacity),
        "items": items
    }
    return instance

# --- I/O helpers ---
def save_instance_json(instance: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(instance, f, indent=2)

def save_instance_csv(instance: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline='', encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "weight", "value"])
        for it in instance["items"]:
            writer.writerow([it["id"], it["weight"], it["value"]])

# --- Batch generator (multiple scales) ---
def generate_batch(
    ns: List[int],
    output_dir: str = "knapsack_instances",
    weight_range: Tuple[int,int] = (1,100),
    value_range: Tuple[int,int] = (1,100),
    capacity_ratio: float = 0.5,
    correlation: Optional[str] = None,
    weight_dist: str = "uniform",
    value_dist: str = "uniform",
    base_seed: int = 42,
    force_overwrite: bool = False
) -> List[Dict]:
    """
    Generate instances for each n in ns and save them.
    Naming: {output_dir}/knapsack_n{n}_seed{seed}.json / .csv
    Returns list of metadata records for each created instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for idx, n in enumerate(ns):
        seed = (base_seed + idx) & 0xFFFFFFFF
        inst = generate_instance(
            n_items=n,
            weight_range=weight_range,
            value_range=value_range,
            capacity_ratio=capacity_ratio,
            correlation=correlation,
            weight_dist=weight_dist,
            value_dist=value_dist,
            seed=seed
        )

        base_name = f"knapsack_n{n}_seed{inst['meta']['seed']}"
        json_path = os.path.join(output_dir, base_name + ".json")
        csv_path  = os.path.join(output_dir, base_name + ".csv")

        if not force_overwrite and os.path.exists(json_path):
            msg = f"exists (skipping): {json_path}"
            # still append record pointing to existing files
            records.append({
                "n": n, "seed": inst['meta']['seed'],
                "json": json_path, "csv": csv_path, "status": "skipped_exists"
            })
            print(msg)
            continue

        save_instance_json(inst, json_path)
        save_instance_csv(inst, csv_path)
        records.append({
            "n": n, "seed": inst['meta']['seed'],
            "json": json_path, "csv": csv_path, "status": "saved"
        })
        # print(f"Saved: n={n} -> {json_path}")
    return records
