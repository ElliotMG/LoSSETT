"""Compare existing and Clenshaw-Curtis latitude weights on real ring fields."""

import argparse
import json
import os
import socket

import numpy as np
import xarray as xr

from h_twice_quadrature_study import clenshaw_curtis_weights
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
    if not np.allclose(np.diff(lat), np.diff(lat)[0], rtol=1e-8, atol=1e-12):
        raise ValueError("Clenshaw-Curtis requires uniform latitudes")
    if not np.allclose(np.abs(lat[[0, -1]]), np.pi/2, rtol=0, atol=1e-12):
        raise ValueError("Clenshaw-Curtis requires pole endpoints")
    u, v, w = (np.asarray(field[name].values, dtype=np.float64) for name in ("u", "v", "w"))
    inputs = np.stack((np.ones_like(w), w, u*u + v*v + w*w))
    origin = np.asarray(initial["origin_rows"])
    h_dir, g_dir = root / "H", root / "G"
    required = np.array(sorted(int(path.stem) for path in h_dir.glob("*.npz")))
    if len(required) != initial["H_required_intermediate_rows"]:
        raise ValueError("incomplete H cache")
    cc = clenshaw_curtis_weights(len(lat)-1)
    areas = {
        "cell_midpoint": _solid_angles(lat, lon),
        "clenshaw_curtis": np.broadcast_to(cc[:, None]*2*np.pi/len(lon), (len(lat), len(lon))),
    }
    results = {}
    for kind, weights in areas.items():
        direct, _ = apply(g_dir, inputs, weights, origin)
        first, _ = apply(h_dir, inputs, weights, required)
        intermediate = np.zeros_like(inputs)
        intermediate[:, required] = first
        twice, _ = apply(h_dir, intermediate, weights, origin)
        data = {}
        for i, name in enumerate(("constant", "radial_velocity", "speed_squared")):
            data[name] = {
                "H_twice_minus_G_max_absolute": float(np.max(np.abs(twice[i] - direct[i]))),
                "H_twice_minus_G_over_G_peak": float(np.max(np.abs(twice[i] - direct[i]))/np.max(np.abs(direct[i]))),
            }
        data["constant"]["H_twice_max_error_vs_one"] = float(np.max(np.abs(twice[0] - 1)))
        data["constant"]["G_max_error_vs_one"] = float(np.max(np.abs(direct[0] - 1)))
        results[kind] = data
    with (root / "quadrature_real.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "grid": args.grid, "origin_rows": origin.tolist(), "results": results,
                   "caveats": "real scalar fields; H-twice minus G is not an absolute error "
                              "except for the known constant field"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
