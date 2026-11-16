"""CLI commands added for the paper: audit, transfer2, burnin, diag2, realdata, paper.

Registered from cli.main(); run as  python -m dtmon <command> -h
"""
from __future__ import annotations
import json
import os
import pandas as pd


def _out(a, name):
    d = os.path.join(a.out, name)
    os.makedirs(d, exist_ok=True)
    return d


def cmd_audit(a):
    from .audit import audit_results, render
    from .experiments import load_results
    S = load_results(a.results)
    A = audit_results(S, alpha=a.alpha)
    out = _out(a, "audit")
    tag = os.path.basename(os.path.normpath(a.results))
    A.to_csv(os.path.join(out, f"audit_{tag}.csv"), index=False)
    print(render(A))
    if a.transfer and os.path.exists(os.path.join(a.transfer, "far.csv")):
        from .audit import audit_transfer
        T = audit_transfer(pd.read_csv(os.path.join(a.transfer, "far.csv")), alpha=a.alpha)
        print("\nnominal-condition false-alarm check (transfer2):")
        print(T[["model", "detector", "far", "far_lo", "far_hi", "flag"]].round(3).to_string(index=False))


def cmd_transfer2(a):
    from .extra import power_transfer
    from .figures_extra import fig_transfer
    out = _out(a, "transfer2")
    fars, pws, sums = [], [], []
    for m in a.models:
        f, p, s = power_transfer(m, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha, n_jobs=a.n_jobs,
                                 levels=tuple(a.levels), cache_dir=os.path.join(a.out, "cache"))
        fars.append(f); pws.append(p); sums.append(s)
        print(f"[transfer2] {m} done", flush=True)
    far, pw, summ = (pd.concat(x, ignore_index=True) for x in (fars, pws, sums))
    far.to_csv(os.path.join(out, "far.csv"), index=False)
    pw.to_csv(os.path.join(out, "power.csv"), index=False)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(fig_transfer(summ, os.path.join(out, "fig_transfer")))
    show = summ[summ.detector.isin(["D4_cusum", "D6_sr", "D7_bocpd", "D8_conformal", "D9_multiscale"])]
    print(show[["model", "shift", "detector", "far", "far_lo", "far_hi", "power_nominal_thr", "power_oracle_thr"]]
          .round(3).to_string(index=False))


def cmd_burnin(a):
    from .extra import burnin_curve
    from .figures_extra import fig_burnin
    out = _out(a, "burnin")
    df = pd.concat([burnin_curve(m, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
                                 cache_dir=os.path.join(a.out, "cache")) for m in a.models], ignore_index=True)
    df.to_csv(os.path.join(out, "burnin_curve.csv"), index=False)
    print(fig_burnin(df, os.path.join(out, "fig_burnin")))
    print(df.round(3).to_string(index=False))


def cmd_diag2(a):
    from .extra import diagnosis_baselines
    from .figures_extra import fig_diag_baselines
    out = _out(a, "diag_baselines")
    S, G = diagnosis_baselines(a.results, n_boot=a.n_boot)
    S.to_csv(os.path.join(out, "diag_baselines.csv"), index=False)
    G.to_csv(os.path.join(out, "diag_subgroups.csv"), index=False)
    if len(S):
        print(fig_diag_baselines(S, os.path.join(out, "fig_diag_baselines")))
    print(S.round(3).to_string(index=False))


def cmd_realdata(a):
    from .realdata import load_series, run_realdata, make_demo_csv
    from .figures_extra import fig_realdata
    out = _out(a, "realdata")
    path = a.csv
    if a.demo:
        path = make_demo_csv(os.path.join(out, "demo_series.csv"))
        print("[realdata] --demo: SYNTHETIC series (pipeline test only, not evidence)")
    if not path:
        raise SystemExit("give --csv FILE (columns id,time,value) or --demo")
    series = load_series(path, a.id_col, a.time_col, a.value_col, min_obs=a.k_burn + 4)
    far, pw, info = run_realdata(series, k_burn=a.k_burn, alpha=a.alpha, n_splits=a.n_splits, seed=a.seed)
    tag = "demo" if a.demo else "real"
    far.to_csv(os.path.join(out, f"{tag}_far.csv"), index=False)
    pw.to_csv(os.path.join(out, f"{tag}_power.csv"), index=False)
    with open(os.path.join(out, f"{tag}_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(info)
    print(fig_realdata(far, pw, os.path.join(out, f"{tag}_fig")))
    print(far.round(3).to_string(index=False))
    print(pw.pivot_table(index="detector", columns=["injected", "magnitude_sigma"], values="power").round(2).to_string())


def register(sub, common):
    sp = sub.add_parser("audit", help="sanity gate: flag degenerate detectors (dead/liberal/no power)")
    common(sp, True)
    sp.add_argument("--transfer", default=None, help="optional results/transfer2 directory")
    sp.set_defaults(fn=cmd_audit)

    sp = sub.add_parser("transfer2", help="H3 with power: false alarms AND power under misspecification")
    common(sp)
    sp.add_argument("--levels", nargs="+", default=["medium"])
    sp.set_defaults(fn=cmd_transfer2)

    sp = sub.add_parser("burnin", help="detection rate vs number of burn-in observations")
    common(sp)
    sp.set_defaults(fn=cmd_burnin)

    sp = sub.add_parser("diag2", help="fault isolation vs baselines, end-to-end, per-subtype (needs main results)")
    common(sp, True)
    sp.set_defaults(fn=cmd_diag2)

    sp = sub.add_parser("realdata", help="semi-synthetic validation on real longitudinal series (csv: id,time,value)")
    common(sp)
    sp.add_argument("--csv", default=None)
    sp.add_argument("--demo", action="store_true", help="use a synthetic demo series to test the pipeline")
    sp.add_argument("--id-col", default="id")
    sp.add_argument("--time-col", default="time")
    sp.add_argument("--value-col", default="value")
    sp.add_argument("--k-burn", type=int, default=5)
    sp.add_argument("--n-splits", type=int, default=50)
    sp.set_defaults(fn=cmd_realdata)
