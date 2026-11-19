#!/usr/bin/env python3
"""CP2K BSE@evGW0@PBE excitation energies for a few molecules: is CP2K a usable GW-BSE label source? (TASK-328)

KNOW-324: CP2K (open source) has GW-BSE for molecules, so new GW-level labels would not need an ADF licence.
Its GW is G0W0/evGW0/evGW, not qsGW, so its labels are a separate fidelity from QM9GWBSE (ADF qsGW-BSE/TZ3P).
For each requested molecule this writes two inputs modelled on the CP2K manual's BSE_H2 example
(all-electron GAPW, PBE, evGW0, full ABBA BSE, aug-cc-pVDZ with aug-cc-pVDZ-RIFIT) for SPIN_CONFIG SINGLET and
TRIPLET, runs them with --cmd, parses the excitation energies and oscillator strengths, and records wall time and
CP2K's BSE memory estimate. Labels, when known, are written next to the result:
  --qm9-heavy 3 6 9     one QM9GWBSE test molecule per heavy-atom count (labels S1-S5 and T1-T5 from --packed)
  --external KEY ...    molecules of the external pool (labels S1-S5 from --labels)
Writes <work-dir>/<name>/{singlet,triplet}.{inp,out} and --report.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
ENERGY_RE = re.compile(r"BSE\|\s+(\d+)\s+(Singlet|Triplet)\s+-ABBA-\s+(-?\d+\.\d+)")
OSC_RE = re.compile(r"BSE\|\s+(\d+)\s+-ABBA-\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s*$")
MEM_RE = re.compile(r"Total peak memory estimate from BSE \[GB\]\s+(\S+)")


def write_input(path: Path, name: str, numbers, positions, spin: str, basis_file: str, n_exc: int, charge: int = 0) -> None:
    pos = np.asarray(positions, dtype=float)
    box = float(np.ptp(pos, axis=0).max() + 12.0)
    coords = "\n".join(f"      {SYMBOL[int(z)]} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(numbers, pos))
    kinds = "\n".join(f"""    &KIND {SYMBOL[int(z)]}
      BASIS_SET ORB    aug-cc-pVDZ
      BASIS_SET RI_AUX aug-cc-pVDZ-RIFIT
      POTENTIAL ALL
    &END KIND""" for z in sorted(set(int(z) for z in numbers)))
    path.write_text(f"""&GLOBAL
  PROJECT {name}_{spin.lower()}
  RUN_TYPE ENERGY
&END GLOBAL
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME {basis_file}
    POTENTIAL_FILE_NAME POTENTIAL
    CHARGE {charge}
    &QS
      METHOD GAPW
      EPS_DEFAULT 1.0E-16
      EPS_PGF_ORB 1.0E-16
    &END QS
    &POISSON
      PERIODIC NONE
      PSOLVER MULTIPOLE
    &END POISSON
    &SCF
      EPS_SCF 1e-7
      MAX_SCF 200
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &WF_CORRELATION
        &RI_RPA
          &GW
            SELF_CONSISTENCY evGW0
            &BSE
              TDA OFF
              SPIN_CONFIG {spin}
              NUM_PRINT_EXC {n_exc}
              ENERGY_CUTOFF_OCC -1
              ENERGY_CUTOFF_EMPTY -1
            &END BSE
          &END GW
        &END RI_RPA
      &END WF_CORRELATION
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC {box:.2f} {box:.2f} {box:.2f}
      PERIODIC NONE
    &END CELL
    &COORD
{coords}
    &END COORD
    &TOPOLOGY
      &CENTER_COORDINATES
      &END CENTER_COORDINATES
    &END TOPOLOGY
{kinds}
  &END SUBSYS
&END FORCE_EVAL
""", encoding="utf-8")


def parse(text: str) -> dict:
    energies = [float(m.group(3)) for m in ENERGY_RE.finditer(text)]
    osc = [float(m.group(5)) for line in text.splitlines() for m in [OSC_RE.search(line)] if m]
    mem = MEM_RE.search(text)
    return {"energies_eV": energies, "osc": osc, "bse_memory_GB": float(mem.group(1)) if mem else None,
            "ok": "PROGRAM ENDED AT" in text and bool(energies)}


def run_one(cmd: str, inp: Path, timeout: int) -> dict:
    out = inp.with_suffix(".out")
    started = time.perf_counter()
    try:
        subprocess.run(f"{cmd} -i {inp.name} -o {out.name}", shell=True, cwd=inp.parent, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "seconds": timeout}
    result = parse(out.read_text(errors="replace") if out.exists() else "")
    result["seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmd", required=True, help='e.g. "mpirun -n 32 cp2k.psmp"')
    parser.add_argument("--basis", required=True, help="CP2K basis file with aug-cc-pVDZ and aug-cc-pVDZ-RIFIT")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--pool", default="data/processed/external_testsets/pool.npz")
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--qm9-heavy", type=int, nargs="*", default=[3, 6, 9])
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--n-exc", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--work-dir", default="runs/cp2k_check")
    parser.add_argument("--report", default="runs/cp2k_check_report.json")
    args = parser.parse_args()
    work = Path(args.work_dir)
    jobs = []  # (name, numbers, positions, labels dict)
    q = np.load(args.packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    off, split = q["atom_offsets"], q["split"].astype(str)
    for heavy in args.qm9_heavy:
        for i in np.flatnonzero(split == "test"):
            z = q["numbers"][off[i]:off[i + 1]]
            if int((z > 1).sum()) == heavy and set(int(v) for v in z) <= {1, 6, 7, 8}:
                labels = {c: float(q["targets"][i, cols.index(c)]) for c in cols if c[0] in "ST" and c.endswith("_eV")}
                jobs.append((f"qm9_{q['molecule_key'][i]}", z, q["positions"][off[i]:off[i + 1]], labels))
                break
    if args.external:
        pool = np.load(args.pool, allow_pickle=False)
        index = {str(k): i for i, k in enumerate(pool["molecule_key"])}
        with open(args.labels, newline="", encoding="utf-8") as stream:
            rows = {r["molecule_key"]: r for r in csv.DictReader(stream)}
        for key in args.external:
            i, poff = index[key], pool["atom_offsets"]
            labels = {f"S{n}_eV": float(rows[key][f"S{n}_eV"]) for n in range(1, 6) if rows[key][f"S{n}_eV"]}
            jobs.append((key.replace(":", "_"), pool["numbers"][poff[i]:poff[i + 1]], pool["positions"][poff[i]:poff[i + 1]], labels))
    report = {"cmd": args.cmd, "method": "BSE@evGW0@PBE, aug-cc-pVDZ / aug-cc-pVDZ-RIFIT, all-electron GAPW, full ABBA",
              "molecules": []}
    for name, z, pos, labels in jobs:
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        entry = {"name": name, "atoms": int(len(z)), "heavy_atoms": int((np.asarray(z) > 1).sum()), "labels": labels}
        for spin in ("SINGLET", "TRIPLET"):
            inp = d / f"{spin.lower()}.inp"
            write_input(inp, name, z, pos, spin, str(Path(args.basis).resolve()), args.n_exc)
            entry[spin.lower()] = run_one(args.cmd, inp, args.timeout)
            print(name, spin, json.dumps({k: v for k, v in entry[spin.lower()].items() if k != "osc"}), flush=True)
        for spin, prefix in (("singlet", "S"), ("triplet", "T")):
            got = entry[spin].get("energies_eV") or []
            ref = [labels.get(f"{prefix}{n}_eV") for n in range(1, 6)]
            pairs = [(g, r) for g, r in zip(got[:5], ref) if r is not None]
            if pairs:
                entry[f"{spin}_minus_label_eV"] = [round(g - r, 4) for g, r in pairs]
        report["molecules"].append(entry)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: m.get(k) for k in ("name", "atoms", "singlet_minus_label_eV", "triplet_minus_label_eV")}
                      for m in report["molecules"]], indent=1))


if __name__ == "__main__":
    main()
