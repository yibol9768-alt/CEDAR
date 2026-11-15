#!/usr/bin/env python3
"""Appendix tables from results_e2/review_analyses.json: paper/tables/t_review_ci.tex and t_c2_genre.tex."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results_e2", "review_analyses.json")))
TAB = os.path.join(ROOT, "paper", "tables")
BB = {"mb": r"\textsc{MB}", "db": r"\textsc{DB}"}
SP = {"test_confirm": r"\textit{Confirm}", "test_confirm2": r"\textit{Confirm-2}"}


def ci(x):
    return f"[{x[0]:+.1f}, {x[1]:+.1f}]"


def main():
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textbf{Uncertainty from parents and from training seeds} (WG difference A$-$B, pp). ``Parents'': 95\% CI resampling "
         r"parent clusters with the five fitted runs fixed (as in the main text). ``Seeds + parents'': each resample also redraws the five "
         r"seeds of A and of B with replacement. Last column: per-seed differences (seed $k$ of A minus seed $k$ of B) that are positive. "
         r"Bold: seeds + parents CI excludes 0.}",
         r"\label{tab:review_ci}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{ll l r c c c}", r"\toprule",
         r"Split & BB & Contrast (A vs.\ B) & $\Delta$ & Parents & Seeds + parents & $>0$ \\", r"\midrule"]
    prev = None
    for k, v in R["contrasts_seed_parent"].items():
        sp, bb, name = k.split("/", 2)
        if prev and sp != prev:
            L.append(r"\midrule")
        prev = sp
        name = name.replace("Spot w/o DTE", r"\textsc{Spot} w/o DTE").replace("Spot", r"\textsc{Spot}").replace("Aug", "Aug.\\ ERM") \
            .replace("GDRO", "Group-DRO").replace("Pair", "Pair consist.").replace("gPD", r"\textsc{Spot}-gPD")
        name = name.replace(r"\textsc{\textsc{Spot}}", r"\textsc{Spot}").replace(r"\textsc{\textsc{Spot}}-gPD", r"\textsc{Spot}-gPD")
        two = ci(v["ci_seeds_and_parents"])
        if v["ci_seeds_and_parents"][0] > 0 or v["ci_seeds_and_parents"][1] < 0:
            two = rf"\textbf{{{two}}}"
        L.append(f"{SP[sp]} & {BB[bb]} & {name} & {v['diff']:+.1f} & {ci(v['ci_parents'])} & {two} & "
                 f"{v['n_seed_diffs_positive']}/{v['n_seeds']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "t_review_ci.tex"), "w").write("\n".join(L) + "\n")

    G = R["confirm2_by_genre"]
    doms = ["raid_books", "raid_news", "raid_recipes", "raid_wiki"]
    rows = [("aug", "Augmented ERM"), ("pair", "ERM + pair consist."), ("spot", r"\textsc{Spot}"),
            ("spot_nodte", r"\textsc{Spot} w/o DTE"), ("gappd_split", r"\textsc{Spot}-gPD")]
    n = G["domains"]
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textbf{\textit{Confirm-2} worst-group accuracy by genre} (\%, mean over 5 seeds, locked threshold). "
         r"Parents per genre: " + ", ".join(f"{d.split('_')[1]} {n[d]}" for d in doms) +
         r". Wikipedia and news also occur in earlier splits; ``w/o Wiki'' drops the Wikipedia parents.}",
         r"\label{tab:c2_genre}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l l cccccc}", r"\toprule",
         r"BB & Method & All & w/o Wiki & Books & News & Recipes & Wiki \\", r"\midrule"]
    for bb in ("mb", "db"):
        for t, nm in rows:
            g = G[f"e2_{bb}_{t}"]
            L.append(f"{BB[bb]} & {nm} & " + " & ".join(f"{g[k]:.1f}" for k in ["all", "no_wiki"] + doms) + r" \\")
        if bb == "mb":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "t_c2_genre.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
