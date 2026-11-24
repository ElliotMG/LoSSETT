"""Verify completion and manifest consistency for the n640 FFT archive."""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    expected_tasks = set(range(49))
    manifests = sorted((root / "manifests").glob("*.json"))
    if {int(path.stem) for path in manifests} != expected_tasks:
        raise ValueError("Missing or unexpected task manifests")
    reports = [json.loads(path.read_text()) for path in manifests]
    identity = reports[0]["identity"]
    if identity["grid_shape"] != [961, 1280]:
        raise ValueError("Unexpected grid shape in archive manifest")
    rows = {}
    hosts = set()
    for path, report in zip(manifests, reports):
        task = int(path.stem)
        if report["array_task_id"] != task or report["identity"] != identity:
            raise ValueError(f"Inconsistent task manifest {path}")
        hosts.add(report["host"])
        expected = list(range(task * 20, min((task + 1) * 20, 961)))
        if [entry["row"] for entry in report["rows"]] != expected:
            raise ValueError(f"Incorrect row allocation in {path}")
        for entry in report["rows"]:
            row = entry["row"]
            archive = root / "kernels" / f"{row:05d}.npz"
            if not archive.is_file() or archive.stat().st_size != entry["bytes"]:
                raise ValueError(f"Missing or unexpected archive size: {archive}")
            rows[row] = entry
    expected_names = {f"{row:05d}.npz" for row in range(961)}
    actual_names = {path.name for path in (root / "kernels").iterdir()}
    if actual_names != expected_names or set(rows) != set(range(961)):
        raise ValueError("Incomplete, duplicate, or unexpected kernel archives")
    reference_rows = sorted(
        row for row, entry in rows.items() if entry["checked_against_serial_reference"]
    )
    if reference_rows != [0, 240, 480, 720, 960]:
        raise ValueError(f"Unexpected serial-reference checks: {reference_rows}")
    summary = {
        "status": "complete; experimental archive",
        "identity": identity,
        "rows": len(rows),
        "task_count": len(reports),
        "hosts": sorted(hosts),
        "archive_bytes": sum(entry["bytes"] for entry in rows.values()),
        "total_build_seconds": sum(entry["build_seconds"] for entry in rows.values()),
        "total_write_seconds": sum(entry["write_seconds"] for entry in rows.values()),
        "total_readback_seconds": sum(entry["readback_seconds"] for entry in rows.values()),
        "max_task_peak_rss_kib": max(report["peak_rss_kib"] for report in reports),
        "serial_reference_rows": reference_rows,
        "recovered_existing_rows": [
            row for row, entry in sorted(rows.items()) if entry.get("reused_existing", False)
        ],
        "validation": (
            "Each task verified finite values and read-back equality on creation; "
            "recovered rows were compared with regenerated kernels at rtol=1e-12 and atol=1e-11; "
            "this summary verifies manifests, filenames and sizes, not persistent checksums. "
            "Timings for recovered rows represent regeneration rather than the original build."
        ),
    }
    with (root / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
