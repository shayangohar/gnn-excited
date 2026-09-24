#!/usr/bin/env python3
"""Stream PubChemQC records from the revived Postgres DB straight into HDF5.

Reads public.b3lyp(cid, data jsonb) with a server-side cursor and writes
schema pubchemqc-st-v1 (units eV):
  HDF5  /m/pqc_<cid>/numbers(i32[N]) positions(f64[N,3]) S_eV(f64[10])
  CSV   molecule_key,inchikey,smiles,natoms,S1_eV..S10_eV,split,status
Keeps neutral singlets only (charge==0, multiplicity==1); transitions are
cm^-1 in the DB and converted with CM_TO_EV. Split = inchikey hash bucket.
"""
from __future__ import annotations

import argparse
import csv
import hashlib

import h5py
import numpy as np
import psycopg2

CM_TO_EV = 1.0 / 8065.54429
N_STATES = 10


def split_for(key: str, val_fraction: float = 0.05) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return "val" if int(digest[:16], 16) % 10_000 < val_fraction * 10_000 else "train"


def parse_record(cid: int, data: dict):
    """Return dict(numbers, positions, S_eV, inchikey, smiles, formula) or (None, reason)."""
    try:
        props = data.get("properties") or {}
        if int(props.get("charge", 0)) != 0:
            return None, "charged"
        if int(props.get("multiplicity", 1)) != 1:
            return None, "non_singlet"
        atoms = data.get("atoms") or {}
        numbers = [int(v) for v in (atoms.get("elements") or {}).get("number", [])]
        flat = [float(v) for v in (atoms.get("coords") or {}).get("3d", [])]
        if not numbers or len(flat) != 3 * len(numbers):
            return None, "bad_geometry"
        if any(z < 1 or z > 100 for z in numbers):
            return None, "bad_element"
        transitions = (data.get("transitions") or {}).get("electronic transitions", [])
        if not isinstance(transitions, list) or len(transitions) < N_STATES:
            return None, "bad_transitions"
        S_eV = np.asarray([float(v) * CM_TO_EV for v in transitions[:N_STATES]], dtype=np.float64)
        if not np.isfinite(S_eV).all():
            return None, "nonfinite_energy"
        if bool((np.diff(S_eV) < 0).any()):
            return None, "unordered"
        return {
            "numbers": np.asarray(numbers, dtype=np.int32),
            "positions": np.asarray(flat, dtype=np.float64).reshape(-1, 3),
            "S_eV": S_eV,
            "inchikey": str(data.get("inchikey") or "").strip(),
            "smiles": str(data.get("smiles") or "").strip(),
            "formula": str(data.get("formula") or "").strip(),
        }, None
    except (ValueError, TypeError, AttributeError) as exc:
        return None, f"parse_error:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-dir", required=True)
    parser.add_argument("--port", type=int, default=55433)
    parser.add_argument("--password", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    header = (["molecule_key", "inchikey", "smiles", "natoms"]
              + [f"S{s}_eV" for s in range(1, N_STATES + 1)]
              + ["split", "status"])
    conn = psycopg2.connect(host=args.socket_dir, port=args.port,
                            user="pgrest", password=args.password, dbname="db")
    scanned = kept = 0
    skips: dict[str, int] = {}
    with conn, conn.cursor(name="pqc_stream") as cur, \
            h5py.File(args.out_h5, "w") as h5, \
            open(args.manifest, "w", newline="", encoding="utf-8") as csvfile:
        root = h5.create_group("m")
        root.attrs["schema"] = "pubchemqc-st-v1"
        root.attrs["units"] = "eV"
        writer = csv.writer(csvfile)
        writer.writerow(header)
        cur.itersize = 10_000
        cur.execute("SELECT cid, data FROM public.b3lyp ORDER BY cid")
        for cid, data in cur:
            scanned += 1
            if scanned % 200_000 == 0:
                print(f"scanned={scanned} kept={kept}", flush=True)
            if args.limit and scanned > args.limit:
                break
            if isinstance(data, str):
                import json
                data = json.loads(data)
            parsed, reason = parse_record(cid, data)
            if parsed is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            if len(parsed["numbers"]) > args.max_atoms:
                skips["too_big"] = skips.get("too_big", 0) + 1
                continue
            key = f"pqc_{cid}"
            grp = root.create_group(key)
            grp.create_dataset("numbers", data=parsed["numbers"])
            grp.create_dataset("positions", data=parsed["positions"])
            grp.create_dataset("S_eV", data=parsed["S_eV"])
            split = split_for(parsed["inchikey"] or key, args.val_fraction)
            grp.attrs["split"] = split
            writer.writerow([key, parsed["inchikey"], parsed["smiles"], str(len(parsed["numbers"]))]
                            + [f"{v:.6f}" for v in parsed["S_eV"]] + [split, "ok"])
            kept += 1
    print(f"done: scanned={scanned} kept={kept} skips={skips}", flush=True)


if __name__ == "__main__":
    main()
