# -*- coding: utf-8 -*-
import csv, statistics

path = r"data\_richards_diagnostics.csv"

rows = []
with open(path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print("total rows:", len(rows))

t_mid_all = []
for r in rows:
    try:
        tm = float(r["t_mid"])
        r2 = float(r["R2"])
        status = r.get("status", "")
        t_mid_all.append((r["entity"], tm, r2, status))
    except Exception as e:
        pass

print("parsed countries:", len(t_mid_all))

# count status
from collections import Counter
print("status counts:", Counter(s for _,_,_,s in t_mid_all))

def report(T_start, label):
    print("\n===== T_start = %s (%s) =====" % (T_start, label))
    # exclude historical polities where t_mid predates T_start (premise of "modern
    # trajectory starting at first-wave feminism" does not apply)
    t_mid_clean = [(n, tm, r2, s) for n, tm, r2, s in t_mid_all if tm >= T_start]
    print("kept (t_mid >= T_start): %d of %d; dropped %d" %
          (len(t_mid_clean), len(t_mid_all), len(t_mid_all) - len(t_mid_clean)))
    dropped = [(n, tm) for n, tm, r2, s in t_mid_all if tm < T_start]
    print("dropped:", dropped)
    # T_end = 2*t_mid - T_start
    data = [(name, tm, 2*tm - T_start) for name, tm, r2, s in t_mid_clean]
    # fastest = smallest T_end
    data_sorted = sorted(data, key=lambda x: x[2])
    t_ends = [d[2] for d in data_sorted]
    t_mids = [d[1] for d in data_sorted]

    def q(seq, p):
        return seq[int(round(p*(len(seq)-1)))]

    print("t_mid:  min=%.1f  p10=%.1f  median=%.1f  p90=%.1f  max=%.1f"
          % (min(t_mids), q(t_mids,0.1), statistics.median(t_mids), q(t_mids,0.9), max(t_mids)))
    print("T_end:  min=%.1f  p10=%.1f  median=%.1f  p90=%.1f  max=%.1f"
          % (min(t_ends), q(t_ends,0.1), statistics.median(t_ends), q(t_ends,0.9), max(t_ends)))
    # duration of phase 1 = T_end - T_start = 2*(t_mid - T_start)
    durs = sorted(2*(tm - T_start) for _, tm, _ in data_sorted)
    print("phase-1 duration (yrs): min=%.0f  median=%.0f  max=%.0f"
          % (durs[0], statistics.median(durs), durs[-1]))
    # years remaining from 2026
    rem = sorted(te - 2026 for te in t_ends)
    print("years from 2026 to T_end: min=%.0f  median=%.0f  max=%.0f"
          % (rem[0], statistics.median(rem), rem[-1]))
    print("fastest 5 (by T_end):")
    for name, tm, te in data_sorted[:5]:
        print("   %-20s t_mid=%.1f  T_end=%.1f" % (name, tm, te))
    print("slowest 5 (by T_end):")
    for name, tm, te in data_sorted[-5:]:
        print("   %-20s t_mid=%.1f  T_end=%.1f" % (name, tm, te))

report(1848, "Seneca Falls, first-wave feminism (authoritative)")
report(1789, "French Revolution / early modern (reference)")
report(1840, "early 19th c. (reference)")
