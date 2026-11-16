"""Command-line interface:  python -m dtmon <command> [options]   (run `python -m dtmon -h`)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd


def _out(args, name):
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    return d


def cmd_pilot(a):
    from .experiments import condition_grid, run_grid
    out = _out(a, "pilot")
    conds = condition_grid(models=a.models, noises=(0.1,), schemes=("dense", "moderate", "sparse", "realistic"), variants=("full",))
    conds += condition_grid(models=a.models, noises=(0.1,), schemes=("moderate",), variants=("no_norm", "open_loop"))
    print(f"[pilot] {len(conds)} conditions, {a.n_patients} patients each -> {out}")
    run_grid(conds, out, n_jobs=a.n_jobs, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
             with_ablation_detectors=True, cache_dir=os.path.join(a.out, "cache"))


def cmd_main(a):
    from .experiments import condition_grid, run_grid
    out = _out(a, f"main_P{a.n_patients}")
    conds = condition_grid(models=a.models, noises=tuple(a.noises), schemes=tuple(a.schemes), variants=("full",))
    print(f"[main] {len(conds)} conditions, {a.n_patients} patients each -> {out}")
    run_grid(conds, out, n_jobs=a.n_jobs, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
             with_ablation_detectors=True, cache_dir=os.path.join(a.out, "cache"))


def cmd_ablation(a):
    from .experiments import condition_grid, run_grid
    out = _out(a, f"ablation_P{a.n_patients}")
    conds = condition_grid(models=a.models, noises=(0.1, 0.2), schemes=("moderate", "realistic", "sparse"),
                           variants=("full", "no_norm", "open_loop"))
    print(f"[ablation] {len(conds)} conditions -> {out}")
    run_grid(conds, out, n_jobs=a.n_jobs, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
             with_ablation_detectors=True, cache_dir=os.path.join(a.out, "cache"))


def cmd_stress(a):
    """Stress tests: measurement noise 5/10/20/40%, missing data, parameter uncertainty, misspecified noise."""
    from .experiments import condition_grid, run_grid
    out = _out(a, f"stress_P{a.n_patients}")
    conds = condition_grid(models=a.models, noises=(0.05, 0.1, 0.2, 0.4), schemes=("realistic",))
    conds += condition_grid(models=a.models, noises=(0.1,), schemes=("moderate",), miss=(0.2, 0.4))
    conds += condition_grid(models=a.models, noises=(0.1,), schemes=("moderate",), cal_sigmas=(0.02, 0.1, 0.2))
    conds += condition_grid(models=a.models, noises=(0.1,), schemes=("moderate",), cv_assumed=(0.5, 2.0))
    print(f"[stress] {len(conds)} conditions -> {out}")
    run_grid(conds, out, n_jobs=a.n_jobs, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
             cache_dir=os.path.join(a.out, "cache"))


def cmd_transfer(a):
    """H3: calibrate at nominal conditions, test false-alarm control under shifted conditions."""
    from .experiments import transfer_far
    out = _out(a, "transfer")
    rows = []
    for model in a.models:
        nominal = dict(noise=0.1, scheme="moderate", cal_sigma=0.05, cv_assumed_scale=1.0, miss=0.0)
        shifts = [dict(nominal, noise=0.2), dict(nominal, noise=0.4), dict(nominal, cv_assumed_scale=0.5),
                  dict(nominal, cv_assumed_scale=2.0), dict(nominal, cal_sigma=0.15), dict(nominal, miss=0.3),
                  dict(nominal, scheme="realistic")]
        df = transfer_far(model, nominal, [nominal] + shifts, n_patients=a.n_patients, seed=a.seed, alpha=a.alpha,
                          cache_dir=os.path.join(a.out, "cache"))
        rows.append(df)
        print(f"[transfer] {model} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(out, "transfer_far.csv"), index=False)
    cols = [c for c in df.columns if c.startswith("test_")]
    piv = df.pivot_table(index="detector", columns=[c for c in cols if df[c].nunique() > 1], values="far")
    print(piv.round(3).to_string())


def cmd_theory(a):
    from .theory import run_all
    out = _out(a, "theory")
    res = run_all(seed=a.seed)
    with open(os.path.join(out, "theory_checks.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    for k, v in res.items():
        print("==", k)
        for r in v:
            print("  ", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()})


def cmd_diagnosis(a):
    from .analysis import diagnosis_table, diagnosis_ablation
    d = a.results
    out = _out(a, "analysis")
    tabs = [diagnosis_table(d, det) for det in ("D4_cusum", "D9_multiscale", "D10_hybrid")]
    T = pd.concat(tabs, ignore_index=True)
    T.drop(columns=["confusion"]).to_csv(os.path.join(out, "diagnosis_table.csv"), index=False)
    print(T.drop(columns=["confusion"]).round(3).to_string())
    ab = diagnosis_ablation(d)
    ab.to_csv(os.path.join(out, "diagnosis_feature_ablation.csv"), index=False)
    print(ab.round(3).to_string())


def cmd_decide(a):
    from .analysis import (add_patient_metrics, publication_decision, hypothesis_scorecard, ablation_effects,
                           variant_effects, summary_table, compare_to_best, BASELINES_CONVENTIONAL, BASELINES_ADVANCED)
    from .experiments import load_patients, load_results
    d = a.results
    out = _out(a, "analysis")
    P = add_patient_metrics(load_patients(d))
    S = load_results(d)
    summary_table(S[S.variant == "full"]).to_csv(os.path.join(out, "summary_by_cell.csv"), index=False)
    dec = publication_decision(P[P.variant == "full"], n_boot=a.n_boot)
    with open(os.path.join(out, "publication_decision.json"), "w") as f:
        json.dump(dec, f, indent=2, default=float)
    print("== Publication decision (candidate vs best baseline at equal false-alarm probability)")
    for det in dec["details"]:
        print(f"  {det['candidate']}: {det['label']}   transferable(D)={det['criterion_D_transferable']}")
        for m, v in det["per_model"].items():
            print(f"     {m}: best baseline={v['best_baseline']}; " + ", ".join(
                f"{k}={v[k]:.3f}" for k in v if k not in ("n_cells", "best_baseline")) + f", cells={v['n_cells']}")
    if "proposals" in dec:
        print("  RED -> proposed next steps:")
        for p in dec["proposals"]:
            print("   -", p)
    for cand in ("D9_multiscale", "D10_hybrid"):
        c = compare_to_best(P[P.variant == "full"], cand, BASELINES_CONVENTIONAL, n_boot=a.n_boot)
        c.to_csv(os.path.join(out, f"{cand}_vs_best_conventional.csv"), index=False)
        c = compare_to_best(P[P.variant == "full"], cand, BASELINES_CONVENTIONAL + BASELINES_ADVANCED, n_boot=a.n_boot)
        c.to_csv(os.path.join(out, f"{cand}_vs_best_any.csv"), index=False)
        c = compare_to_best(P[P.variant == "full"], cand, BASELINES_CONVENTIONAL + BASELINES_ADVANCED, n_boot=a.n_boot,
                            select="cell")
        c.to_csv(os.path.join(out, f"{cand}_vs_oracle_cell_best.csv"), index=False)
    ae = ablation_effects(P)
    ae.to_csv(os.path.join(out, "ablation_detector_effects.csv"), index=False)
    ve = variant_effects(P)
    ve.to_csv(os.path.join(out, "ablation_variant_effects.csv"), index=False)
    sc = hypothesis_scorecard(d)
    sc.to_csv(os.path.join(out, "hypothesis_scorecard.csv"), index=False)
    print("\n== Hypothesis scorecard")
    print(sc.round(3).to_string(index=False))


def cmd_figures(a):
    from .figures import make_all
    fig_dir = os.path.join(a.out, "figures")
    res = make_all(a.results, fig_dir)
    for k, v in res.items():
        print(k, v)


def cmd_all(a):
    cmd_pilot(a)
    a.results = os.path.join(a.out, "pilot")
    cmd_theory(a)
    cmd_decide(a)
    cmd_diagnosis(a)
    cmd_figures(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dtmon", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, results=False):
        sp.add_argument("--out", default="results", help="output root directory")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--alpha", type=float, default=0.05, help="target false-alarm probability")
        sp.add_argument("--n-patients", type=int, default=100, help="virtual patients per cohort (100/500/1000)")
        sp.add_argument("--n-jobs", type=int, default=1)
        sp.add_argument("--models", nargs="+", default=["glucose", "tumor"])
        if results:
            sp.add_argument("--results", required=True, help="directory with summary_*/patients_* files")
        sp.add_argument("--n-boot", type=int, default=500)

    for name, fn, res in [("pilot", cmd_pilot, False), ("main", cmd_main, False), ("ablation", cmd_ablation, False),
                          ("stress", cmd_stress, False), ("transfer", cmd_transfer, False),
                          ("theory", cmd_theory, False), ("diagnosis", cmd_diagnosis, True),
                          ("decide", cmd_decide, True), ("figures", cmd_figures, True), ("all", cmd_all, False)]:
        sp = sub.add_parser(name)
        common(sp, res)
        if name == "main":
            sp.add_argument("--noises", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4])
            sp.add_argument("--schemes", nargs="+", default=["dense", "moderate", "sparse", "realistic"])
        sp.set_defaults(fn=fn)
    from .cli_paper import register
    register(sub, common)
    a = p.parse_args(argv)
    t0 = time.time()
    a.fn(a)
    print(f"[{a.cmd}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
