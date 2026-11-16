"""Convert ebos_data_breast.xlsx (Zenodo 10607753) to long csv id,time,value (vehicle mice, tumor volume mm3)."""
import sys, pandas as pd, numpy as np
src, dst = sys.argv[1], sys.argv[2]
d = pd.read_excel(src, header=None)
VOL = range(11, 36)            # tumor-volume columns (bioluminescence starts at col 36)
rows, days = [], None
for _, r in d.iterrows():
    if isinstance(r[1], str) and r[1].startswith("Days Post"):
        days = {c: r[c] for c in VOL if pd.notna(r[c])}; exp = r[0] if pd.notna(r[0]) else exp if 'exp' in dir() else ''
        continue
    if days is None or pd.isna(r[10]) or not isinstance(r[1], str): continue
    if r[1].strip().lower() != "vehicle": continue
    for c, t in days.items():
        v = pd.to_numeric(r[c], errors="coerce")
        if pd.notna(v) and v > 0: rows.append((str(r[10]).strip(), float(t), float(v), r[1]))
o = pd.DataFrame(rows, columns=["id", "time", "value", "group"])
o[["id", "time", "value"]].to_csv(dst, index=False)
n = o.groupby("id").size()
print("mice", n.size, "obs/mouse min/median/max", n.min(), n.median(), n.max())
print("mice with >=7 obs:", (n >= 7).sum(), "| >=6:", (n >= 6).sum(), "| >=5:", (n >= 5).sum())
