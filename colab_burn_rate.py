"""Compute-unit burn-rate calibration log for Colab.

Colab does not show the A100 cost per hour before a runtime is connected, so the rate has to be
measured. This module records the two balance readings around a timed workload, derives units per
hour, appends the record to a JSON log on mounted Drive, and converts a planned hour budget into
units at the measured rate.

It reads nothing from Colab automatically. The balance figures are typed in by whoever runs the
calibration, because the balance is only shown in the Colab user interface.

Command line:

    python colab_burn_rate.py record --log /content/drive/MyDrive/compliance-molgen/burn_rate.json \
        --accelerator A100 --units-before 200 --units-after 196.4 --minutes 18
    python colab_burn_rate.py budget --log ... --units-available 200 --planned-hours 40
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
from typing import List, Optional


def _load(log_path: str) -> List[dict]:
    path = pathlib.Path(log_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write(log_path: str, records: List[dict]) -> None:
    path = pathlib.Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    os.replace(tmp, path)


def record(
    log_path: str,
    accelerator: str,
    units_before: float,
    units_after: float,
    minutes: float,
    note: str = "",
) -> dict:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    spent = units_before - units_after
    if spent < 0:
        raise ValueError("units_after is higher than units_before, check the readings")
    hours = minutes / 60.0
    entry = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "accelerator": accelerator,
        "units_before": units_before,
        "units_after": units_after,
        "units_spent": round(spent, 4),
        "minutes": minutes,
        "units_per_hour": round(spent / hours, 4),
        "note": note,
    }
    records = _load(log_path)
    records.append(entry)
    _write(log_path, records)
    return entry


def measured_rate(log_path: str, accelerator: str) -> Optional[float]:
    """Most recent measured units per hour for an accelerator, or None if never measured."""
    for entry in reversed(_load(log_path)):
        if entry["accelerator"] == accelerator:
            return entry["units_per_hour"]
    return None


def budget(
    log_path: str,
    accelerator: str,
    units_available: float,
    planned_hours: float,
) -> dict:
    rate = measured_rate(log_path, accelerator)
    out = {
        "accelerator": accelerator,
        "units_available": units_available,
        "planned_hours": planned_hours,
        "units_per_hour": rate,
    }
    if rate is None or rate <= 0:
        out["status"] = "no measurement yet, run the calibration first"
        return out
    out["hours_affordable"] = round(units_available / rate, 2)
    out["units_for_plan"] = round(planned_hours * rate, 2)
    out["fits"] = out["units_for_plan"] <= units_available
    out["shortfall_units"] = round(max(0.0, out["units_for_plan"] - units_available), 2)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="log one calibration measurement")
    rec.add_argument("--log", required=True)
    rec.add_argument("--accelerator", default="A100")
    rec.add_argument("--units-before", type=float, required=True)
    rec.add_argument("--units-after", type=float, required=True)
    rec.add_argument("--minutes", type=float, required=True)
    rec.add_argument("--note", default="")

    bud = sub.add_parser("budget", help="convert a planned hour budget into units")
    bud.add_argument("--log", required=True)
    bud.add_argument("--accelerator", default="A100")
    bud.add_argument("--units-available", type=float, required=True)
    bud.add_argument("--planned-hours", type=float, required=True)

    args = parser.parse_args()
    if args.command == "record":
        print(json.dumps(record(args.log, args.accelerator, args.units_before,
                                args.units_after, args.minutes, args.note), indent=2))
    else:
        print(json.dumps(budget(args.log, args.accelerator, args.units_available,
                                args.planned_hours), indent=2))


if __name__ == "__main__":
    _main()
