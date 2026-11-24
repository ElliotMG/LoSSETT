"""Repeat cached scalar H and G ring timings, alternating execution order."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from h_twice_rings_benchmark import DATA, ROOT, apply
from lossett.calc.compute_group_increments import _solid_angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    args = parser.parse_args()
    root = ROOT / f"h_twice_scalar_rings_{args.grid}_20261124"
    initial = json.loads((root / "results.json").read_text())
    with xr.open_dataset(DATA / f"filtered_{args.grid}.nc", decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    weights = _solid_angles(lat, lon)
    u, v, w = (np.asarray(field[name].values, dtype=np.float64) for name in ("u", "v", "w"))
    inputs = np.stack((np.ones_like(w), w, u*u + v*v + w*w))
    origin = np.asarray(initial["origin_rows"])
    h_dir, g_dir = root / "H", root / "G"
    required = np.array(sorted(int(path.stem) for path in h_dir.glob("*.npz")))
    if len(required) != initial["H_required_intermediate_rows"]:
        raise ValueError("incomplete H cache")
    ref_g = np.load(root / "direct_G.npy")
    ref_h = np.load(root / "twice_H.npy")

    def run_g():
        start = time.perf_counter()
        result, parts = apply(g_dir, inputs, weights, origin)
        total = time.perf_counter()-start
        np.testing.assert_array_equal(result, ref_g)
        return {"total_seconds": total, "pass": parts}

    def run_h():
        start = time.perf_counter()
        first, first_parts = apply(h_dir, inputs, weights, required)
        middle = np.zeros_like(inputs)
        middle[:, required] = first
        result, second_parts = apply(h_dir, middle, weights, origin)
        total = time.perf_counter()-start
        np.testing.assert_array_equal(result, ref_h)
        return {"total_seconds": total, "first_pass": first_parts, "second_pass": second_parts}

    run_g()
    run_h()
    repeats = []
    for iteration in range(5):
        order = ("G", "H") if iteration % 2 == 0 else ("H", "G")
        record = {"order": order}
        for item in order:
            record[item] = run_g() if item == "G" else run_h()
        repeats.append(record)
    summary = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "grid": args.grid, "cache": "warm/unknown; one warm-up; alternating order",
        "G_seconds": [record["G"]["total_seconds"] for record in repeats],
        "H_twice_seconds": [record["H"]["total_seconds"] for record in repeats],
        "repetitions": repeats,
    }
    summary["G_median_seconds"] = float(np.median(summary["G_seconds"]))
    summary["H_twice_median_seconds"] = float(np.median(summary["H_twice_seconds"]))
    with (root / "repeats.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
