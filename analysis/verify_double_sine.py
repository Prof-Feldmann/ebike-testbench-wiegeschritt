#!/usr/bin/env python3
"""Reproduce the headline result: one actuator stroke per two pedal revolutions.

Usage:  python verify_double_sine.py ../data
"""
import sys, glob, os, gzip
import numpy as np

FS = 20.0  # Hz, sampling rate of the published extract


def load(path):
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        return hdr, np.loadtxt(fh, delimiter=",")


def rising_edges(x):
    return np.where(np.diff((x > 0.5).astype(np.int8)) == 1)[0]


def main(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "signals_*.csv.gz")))
    if not files:
        sys.exit("no signal files found in %s" % data_dir)

    all_rps, total_strokes, total_rev = [], 0, 0
    print("%-8s %9s %9s %9s %9s" % ("segment", "strokes", "revs", "rev/stroke", "cadence"))
    for f in files:
        hdr, X = load(f)
        col = {n: i for i, n in enumerate(hdr)}
        cad = X[:, col["cadence_rpm"]]
        run = cad > 5
        if run.sum() < 100:
            print("%-8s %9s" % (os.path.basename(f)[-10:-7], "idle"))
            continue
        # cumulative pedal revolutions from cadence
        rev_cum = np.cumsum(cad) / 60.0 / FS
        on = rising_edges(X[:, col["cyl_right_on"]])
        rps = np.diff(rev_cum[on])
        rps = rps[(rps > 1) & (rps < 4)]          # drop gaps and standstill
        all_rps.append(rps)
        total_strokes += len(on)
        total_rev += rev_cum[-1]
        print("%-8s %9d %9.0f %9.3f %9.1f"
              % (os.path.basename(f)[-10:-7], len(on), rev_cum[-1],
                 rev_cum[-1] / len(on), cad[run].mean()))

    # Aggregate ratio: robust against the 50 ms edge quantisation of the 20 Hz extract
    print()
    print("strokes (all segments)     : %d" % total_strokes)
    print("pedal revolutions          : %.0f" % total_rev)
    print("aggregate rev/stroke       : %.4f" % (total_rev / total_strokes))

    # Per-stroke distribution: taken from the derived table, computed at the
    # original 100 Hz. At 20 Hz the edge timestamps are quantised to 50 ms,
    # which widens the per-stroke distribution without shifting the aggregate.
    tbl = os.path.join(data_dir, "strokes_2021-06-30.csv")
    if os.path.exists(tbl):
        d = np.genfromtxt(tbl, delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
        v = d["revolutions_since_previous"]
        v = v[(v > 1) & (v < 4)]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        print("per-stroke median (100 Hz) : %.4f  (IQR %.4f, n=%d)"
              % (med, q3 - q1, len(v)))
    else:
        med = total_rev / total_strokes
        print("per-stroke table not found; using aggregate")

    print("expected by the control law: 2.0000")
    dev = abs(med - 2.0)
    print()
    print("deviation from 2.0000      : %.4f  ->  %s"
          % (dev, "PASS" if dev < 0.01 else "CHECK"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data")
