import os
import json
from decimal import Decimal, ROUND_HALF_UP


def r1(x, digits=1):
    if x is None:
        return "--"
    q = Decimal("10") ** (-digits)
    val = (Decimal(str(x)) * Decimal("100")).quantize(q, rounding=ROUND_HALF_UP)
    return f"{val:.{digits}f}"


def r_raw(x, digits=3):
    if x is None:
        return "--"
    q = Decimal("10") ** (-digits)
    val = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)
    return f"{val:.{digits}f}"


def pct(x, digits=1):
    return r1(x, digits=digits)


def pct_ms(d, digits=1, bold=False):
    if d is None or d.get("mean") is None:
        return "--"
    s = f"{r1(d['mean'], digits)}$\\pm${r1(d['std'], digits)}"
    return f"\\textbf{{{s}}}" if bold else s


def best_keys(keys, agg, getter, higher_better=True, tol=0.0005):
    vals = {}
    for k in keys:
        if k in agg:
            d = getter(agg[k])
            if d is not None and d.get("mean") is not None:
                vals[k] = float(d["mean"])
    if not vals:
        return set()
    opt = max(vals.values()) if higher_better else min(vals.values())
    if higher_better:
        return {k for k, v in vals.items() if v >= opt - tol}
    return {k for k, v in vals.items() if v <= opt + tol}


def generate_tables():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res_dir = os.path.join(root, "results_v3")
    tab_dir = os.path.join(root, "paper", "tables")
    os.makedirs(tab_dir, exist_ok=True)

    S = json.load(open(os.path.join(res_dir, "master_evaluation_summary_v3.json")))
    G = json.load(open(os.path.join(res_dir, "nsd_v3_theorem2_and_geometry.json")))

    pre = S["pretrained_and_probes"]
    agg = S["trained_multi_seed_agg"]
    sig = S["paired_significance_tests"]
    holm = S.get("holm_bonferroni_primary_family", {}).get("holm_adjusted_p", {})
    bridge = S["attribution_bridge_2x2"]
    crc = S["few_shot_human_anchor_crc"]

    mb_rows = [
        ("mb_v3_unedited_erm", "Unedited ERM ($E_0$)$^{\\S}$"),
        ("mb_v3_augmented_erm", "Augmented ERM"),
        ("mb_v3_joint_8class_ce", "Joint 8-Class CE"),
        ("mb_v3_multitask_2head", "Multi-Task 3-Head"),
        ("mb_v3_erm_paired_consist", "ERM + Paired Align."),
        ("mb_v3_erm_group_dro", "ERM + Group-DRO"),
        ("mb_v3_supcon_orth_disentangle", "SupCon + Orth."),
        ("mb_v3_strong_combined_no_nsd", "Strong Combined (w/o NSD)"),
        ("mb_v3_cedar_split_op", "\\textbf{\\textsc{CEDAR} Split-Op (Ours)}"),
        ("mb_v3_cedar_full800", "\\textsc{CEDAR} Full-800"),
    ]
    # Primary head-to-head comparison keys (excluding pooled Full-800 reference so primary Split-Op is judged fairly against split-compliant baselines!)
    mb_primary_keys = [k for k, _ in mb_rows if k != "mb_v3_cedar_full800"]

    # =========================================================================
    # TABLE 1: Controlled Symmetric Recovery (test_id) & Pristine Untouched Confirmation (test_confirm)
    # True unscaled \scriptsize (7pt) via tabular* — ZERO \resizebox!
    # =========================================================================
    b_f2 = best_keys(mb_primary_keys, agg, lambda a: a["id_recoverability"]["E2"]["fpr_e0cal"], higher_better=False)
    b_e0 = best_keys(mb_primary_keys, agg, lambda a: a["id_recoverability"]["E0"]["bal_acc"], higher_better=True)
    b_e2 = best_keys(mb_primary_keys, agg, lambda a: a["id_recoverability"]["E2"]["bal_acc"], higher_better=True)
    b_e3 = best_keys(mb_primary_keys, agg, lambda a: a["id_recoverability"]["E3"]["bal_acc"], higher_better=True)
    b_idwg = best_keys(mb_primary_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)
    b_cfauc = best_keys(mb_primary_keys, agg, lambda a: a["confirm_auroc"], higher_better=True)
    b_cfwg = best_keys(mb_primary_keys, agg, lambda a: a["confirm_wg_E0_E3"], higher_better=True)
    b_cfug = best_keys(mb_primary_keys, agg, lambda a: a["confirm_unseen_genres_wg"], higher_better=True)
    b_cfcol = best_keys(mb_primary_keys, agg, lambda a: a["confirm_by_register"]["colloquial_humanized"]["bal_acc"], higher_better=True)

    t1 = [
        "\\begin{table}[t]",
        "\\caption{Symmetric recovery on \\texttt{test\\_id} ($2{,}000$ passages, $200$ parents) and untouched pristine confirmation \\texttt{test\\_confirm} ($1{,}280$ passages, $160$ unseen parents: \\textsc{Granite-3.1-8B} drafts + \\textsc{Phi-4-mini} bidirectional edits + $80$ unseen-genre human families) at locked source-stratified \\texttt{val\\_calib} thresholds ($\\alpha_{\\mathrm{fpr}}{=}0.05$; 5-seed mean$\\pm$SD \\%, \\texttt{ddof=0} on \\textsc{ModernBERT-large}; $^{\\S}$$1{,}600$ $E_0$ passages cycled to match steps; bold marks best primary split-compliant controlled detector within $0.05$\\,pp; pooled \\textsc{CEDAR} Full-800 is shown below as an unsplit upper reference).}",
        "\\label{tab:main_id}",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{1.1pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Detector}} & \\multicolumn{5}{c}{\\textbf{Controlled \\texttt{test\\_id} ($200$ Parents)}} & \\multicolumn{4}{c}{\\textbf{Pristine \\texttt{test\\_confirm} ($160$ Unseen Parents)}} \\\\",
        "\\cmidrule(lr){2-6} \\cmidrule(lr){7-10}",
        " & \\textbf{$E_2$FPR$_{\\tau_{E_0}}\\downarrow$} & \\textbf{Bal$E_0\\uparrow$} & \\textbf{Bal$E_2\\uparrow$} & \\textbf{Bal$E_3\\uparrow$} & \\textbf{WG$_{0..3}\\uparrow$} & \\textbf{AUC$\\uparrow$} & \\textbf{WG$_{0..3}\\uparrow$} & \\textbf{UnseenWG$\\uparrow$} & \\textbf{ColloqBal$\\uparrow$} \\\\",
        "\\midrule",
        "\\multicolumn{10}{@{}l}{\\emph{Zero-shot curvature (\\textsc{Qwen2.5-3B}) \\& external pretrained detectors ($^{\\dagger}$)}} \\\\",
    ]

    for p_key, p_lbl in [
        ("fast_detectgpt_qwen3b", "\\textsc{Fast-DetectGPT}"),
        ("binoculars_qwen3b", "\\textsc{Binoculars}"),
        ("hc3_roberta_detector", "\\textsc{HC3-RoBERTa}$^{\\dagger}$"),
        ("radar_vicuna7b", "\\textsc{RADAR-7B}$^{\\dagger}$"),
        ("desklib_raid_v1", "\\textsc{Desklib-v1}$^{\\dagger}$"),
    ]:
        p = pre[p_key]
        r = p["test_id_recoverability"]
        tc = p["test_confirm"]
        t1.append(
            f"{p_lbl} & {pct(r['E2']['fpr_at_tau_e0_5'])} & {pct(r['E0']['bal_acc'])} & "
            f"{pct(r['E2']['bal_acc'])} & {pct(r['E3']['bal_acc'])} & {pct(p['test_id_worst_group_acc_E0_E3'])} & "
            f"{pct(tc['overall_auroc'])} & {pct(tc['worst_group_acc_E0_E3'])} & "
            f"{pct(tc['unseen_genres_wg_E0_E3'])} & {pct(tc['by_register']['colloquial_humanized']['bal_acc_tau_all5'])} \\\\"
        )

    t1.append("\\midrule")
    t1.append("\\multicolumn{10}{@{}l}{\\emph{Controlled 5-seed neural suite (\\textsc{ModernBERT-large}, bidirectional $800$-parent \\texttt{train\\_v2})}} \\\\")
    for m_key, m_lbl in mb_rows:
        if m_key == "mb_v3_cedar_split_op":
            t1.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        rc = a["id_recoverability"]
        t1.append(
            f"{m_lbl} & {pct_ms(rc['E2']['fpr_e0cal'], bold=(m_key in b_f2))} & "
            f"{pct_ms(rc['E0']['bal_acc'], bold=(m_key in b_e0))} & "
            f"{pct_ms(rc['E2']['bal_acc'], bold=(m_key in b_e2))} & "
            f"{pct_ms(rc['E3']['bal_acc'], bold=(m_key in b_e3))} & "
            f"{pct_ms(a['id_wg_E0_E3'], bold=(m_key in b_idwg))} & "
            f"{pct_ms(a['confirm_auroc'], bold=(m_key in b_cfauc))} & "
            f"{pct_ms(a['confirm_wg_E0_E3'], bold=(m_key in b_cfwg))} & "
            f"{pct_ms(a['confirm_unseen_genres_wg'], bold=(m_key in b_cfug))} & "
            f"{pct_ms(a['confirm_by_register']['colloquial_humanized']['bal_acc'], bold=(m_key in b_cfcol))} \\\\"
        )
    t1.extend(["\\bottomrule", "\\end{tabular*}%", "}", "\\end{table}"])
    with open(os.path.join(tab_dir, "tab1_main_recoverability.tex"), "w") as f:
        f.write("\n".join(t1) + "\n")

    # =========================================================================
    # TABLE 2: Crossed-OOD (AR + Masked Diffusion) & Four External Wild Benchmarks
    # =========================================================================
    b_ar_a = best_keys(mb_primary_keys, agg, lambda a: a["ood_ar_auroc"], higher_better=True)
    b_ar_b = best_keys(mb_primary_keys, agg, lambda a: a["ood_ar_mean_bal"], higher_better=True)
    b_ll_a = best_keys(mb_primary_keys, agg, lambda a: a["ood_llada_auroc"], higher_better=True)
    b_ll_m = best_keys(mb_primary_keys, agg, lambda a: a["ood_llada_len120_220_auroc"], higher_better=True)
    b_esl = best_keys(mb_primary_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    b_bm = best_keys(mb_primary_keys, agg, lambda a: a["beemo_auroc"], higher_better=True)
    b_msp = best_keys(mb_primary_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    b_msa = best_keys(mb_primary_keys, agg, lambda a: a["mixset_all_auroc"], higher_better=True)
    b_mg = best_keys(mb_primary_keys, agg, lambda a: a["mage_wild_auroc"], higher_better=True)

    t2 = [
        "\\begin{table}[t]",
        "\\caption{Zero-shot transfer to Crossed-OOD splits (unseen \\textsc{OLMo-2-7B} AR and \\textsc{LLaDA-8B} masked diffusion, reporting both full $900$-passage and common-support $[120,220]$w $137$-parent AUROC) and four external benchmarks (5-seed mean$\\pm$SD \\%, \\texttt{ddof=0}; bold marks best primary split-compliant controlled detector).}",
        "\\label{tab:ood_external}",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{1.05pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Detector}} & \\multicolumn{2}{c}{\\textbf{OOD-AR (\\textsc{OLMo-2})}} & \\multicolumn{2}{c}{\\textbf{OOD-Diff (\\textsc{LLaDA})}} & \\textbf{ESL $E_1$} & \\textbf{Beemo} & \\multicolumn{2}{c}{\\textbf{MixSet AUROC$\\uparrow$}} & \\textbf{MAGE} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){8-9}",
        " & \\textbf{AUC$\\uparrow$} & \\textbf{Bal$\\uparrow$} & \\textbf{Full$\\uparrow$} & \\textbf{120--220w$\\uparrow$} & \\textbf{FPR$\\downarrow$} & \\textbf{AUC$\\uparrow$} & \\textbf{Pure(620)} & \\textbf{All(720)} & \\textbf{AUC$\\uparrow$} \\\\",
        "\\midrule",
    ]
    for p_key, p_lbl in [
        ("fast_detectgpt_qwen3b", "\\textsc{Fast-DetectGPT}"),
        ("binoculars_qwen3b", "\\textsc{Binoculars}"),
        ("hc3_roberta_detector", "\\textsc{HC3-RoBERTa}$^{\\dagger}$"),
        ("radar_vicuna7b", "\\textsc{RADAR-7B}$^{\\dagger}$"),
        ("desklib_raid_v1", "\\textsc{Desklib-v1}$^{\\dagger}$"),
    ]:
        p = pre[p_key]
        t2.append(
            f"{p_lbl} & {pct(p['test_ood_ar']['overall_auroc'])} & {pct(p['test_ood_ar']['mean_bal_acc'])} & "
            f"{pct(p['test_ood_llada']['overall_auroc'])} & {pct(p['test_ood_llada']['len120_220_auroc'])} & "
            f"{pct(p['external_esl_fairness']['esl_E1_polished_fpr_at_tau_all5'])} & "
            f"{pct(p['external_beemo']['overall_auroc'])} & "
            f"{pct(p['external_mixset']['pure_origin_auroc'])} & {pct(p['external_mixset']['all_720_auroc'])} & "
            f"{pct(p['external_mage_wild']['auroc'])} \\\\"
        )
    t2.append("\\midrule")
    for m_key, m_lbl in mb_rows:
        if m_key == "mb_v3_cedar_split_op":
            t2.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t2.append(
            f"{m_lbl} & {pct_ms(a['ood_ar_auroc'], bold=(m_key in b_ar_a))} & "
            f"{pct_ms(a['ood_ar_mean_bal'], bold=(m_key in b_ar_b))} & "
            f"{pct_ms(a['ood_llada_auroc'], bold=(m_key in b_ll_a))} & "
            f"{pct_ms(a['ood_llada_len120_220_auroc'], bold=(m_key in b_ll_m))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in b_esl))} & "
            f"{pct_ms(a['beemo_auroc'], bold=(m_key in b_bm))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in b_msp))} & "
            f"{pct_ms(a['mixset_all_auroc'], bold=(m_key in b_msa))} & "
            f"{pct_ms(a['mage_wild_auroc'], bold=(m_key in b_mg))} \\\\"
        )
    t2.extend(["\\bottomrule", "\\end{tabular*}%", "}", "\\end{table}"])
    with open(os.path.join(tab_dir, "tab2_ood_and_external.tex"), "w") as f:
        f.write("\n".join(t2) + "\n")

    # =========================================================================
    # TABLE 3: Mechanistic Ablations (N_fit=600), 2x2 Data x Recipe Bridge, & DeBERTa-v3-large (5 Seeds)
    # =========================================================================
    ab_rows = [
        ("mb_v3_cedar_split_op", "\\textbf{\\textsc{CEDAR} Split-Op ($r{=}24, \\lambda_{\\mathrm{prot}}{=}0.5$, 5s)}"),
        ("mb_v3_strong_combined_fit600_no_nsd", "\\quad w/o \\textsc{NSD} ($r{=}0$, matched $N_{\\mathrm{fit}}{=}600$)"),
        ("mb_v3_ablation_ema_only", "\\quad w/ Online EMA \\textsc{NSD} Only (no \\texttt{train\\_op})"),
        ("mb_v3_ablation_unprotected_nsd", "\\quad w/ Unprotected \\textsc{NSD} ($\\lambda_{\\mathrm{prot}}{=}0$)"),
        ("mb_v3_ablation_no_dte", "\\quad w/o Kinematics (\\textsc{DTE})"),
        ("mb_v3_ablation_param_matched_mlp", "\\quad w/ Param-Matched Doc-MLP ($122$K)"),
        ("mb_v3_ablation_unweighted_pair", "\\quad w/ Unweighted Alignment ($w_{\\mathrm{ret}}\\equiv 1$)"),
        ("mb_v3_ablation_no_xmargin", "\\quad w/o Within-Parent Margin ($\\mathcal{L}_{\\mathrm{xm}}{=}0$)"),
    ]
    ab_keys = [k for k, _ in ab_rows]
    ab_cwg = best_keys(ab_keys, agg, lambda a: a["confirm_wg_E0_E3"], higher_better=True)
    ab_iwg = best_keys(ab_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)
    ab_ll = best_keys(ab_keys, agg, lambda a: a["ood_llada_len120_220_auroc"], higher_better=True)
    ab_ms = best_keys(ab_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    ab_bm = best_keys(ab_keys, agg, lambda a: a["beemo_auroc"], higher_better=True)
    ab_esl = best_keys(ab_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    ab_rc = best_keys(ab_keys, agg, lambda a: a["rc_margin_aurc"], higher_better=False)

    br_rows = [
        ("bridge_v1data_aug_erm", "\\texttt{v1} Unidir.\\ Data + Augmented ERM"),
        ("bridge_v1data_v1cedar", "\\texttt{v1} Unidir.\\ Data + Legacy \\textsc{CEDAR-v1}"),
        ("bridge_v1data_v2cedar", "\\texttt{v1} Unidir.\\ Data + \\textsc{CEDAR} Split-Op"),
    ]

    db_rows = [
        ("db_v3_unedited_erm", "\\textsc{DeBERTa-v3}: Unedited ERM$^{\\S}$ (5s)"),
        ("db_v3_augmented_erm", "\\textsc{DeBERTa-v3}: Augmented ERM (5s)"),
        ("db_v3_erm_paired_consist", "\\textsc{DeBERTa-v3}: ERM + Paired Align.\\ (5s)"),
        ("db_v3_supcon_orth_disentangle", "\\textsc{DeBERTa-v3}: SupCon + Orth.\\ (5s)"),
        ("db_v3_strong_combined_no_nsd", "\\textsc{DeBERTa-v3}: Strong w/o \\textsc{NSD} (5s)"),
        ("db_v3_cedar_split_op", "\\textbf{\\textsc{DeBERTa-v3}: \\textsc{CEDAR} Split-Op (5s)}"),
    ]
    db_keys = [k for k, _ in db_rows]
    db_cwg = best_keys(db_keys, agg, lambda a: a["confirm_wg_E0_E3"], higher_better=True)
    db_iwg = best_keys(db_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)
    db_ll = best_keys(db_keys, agg, lambda a: a["ood_llada_len120_220_auroc"], higher_better=True)
    db_ms = best_keys(db_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    db_bm = best_keys(db_keys, agg, lambda a: a["beemo_auroc"], higher_better=True)
    db_esl = best_keys(db_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    db_rc = best_keys(db_keys, agg, lambda a: a["rc_margin_aurc"], higher_better=False)

    t3 = [
        "\\begin{table}[t]",
        "\\caption{Mechanistic ablations at matched $N_{\\mathrm{fit}}{=}600$ (3 seeds; reference \\textsc{CEDAR} 5 seeds), $2{\\times}2$ unidirectional-vs.-bidirectional data $\\times$ recipe bridge, and 5-seed \\textsc{DeBERTa-v3-large} verification.}",
        "\\label{tab:ablations}",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{1.6pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Variant / Attribution Cell / Backbone} & \\textbf{ConfWG$\\uparrow$} & \\textbf{ID WG$\\uparrow$} & \\textbf{LLaDA$_{120..220}\\uparrow$} & \\textbf{MixSet$\\uparrow$} & \\textbf{Beemo$\\uparrow$} & \\textbf{ESL FPR$\\downarrow$} & \\textbf{AURC$_m\\downarrow$} \\\\",
        "\\midrule",
        "\\multicolumn{8}{@{}l}{\\textsc{ModernBERT-large} \\emph{mechanistic \\& leave-one-out controls ($N_{\\mathrm{fit}}{=}600$)}} \\\\",
    ]
    for m_key, m_lbl in ab_rows:
        if m_key == "mb_v3_cedar_split_op":
            t3.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t3.append(
            f"{m_lbl} & {pct_ms(a['confirm_wg_E0_E3'], bold=(m_key in ab_cwg))} & "
            f"{pct_ms(a['id_wg_E0_E3'], bold=(m_key in ab_iwg))} & "
            f"{pct_ms(a['ood_llada_len120_220_auroc'], bold=(m_key in ab_ll))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in ab_ms))} & "
            f"{pct_ms(a['beemo_auroc'], bold=(m_key in ab_bm))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in ab_esl))} & "
            f"{pct_ms(a['rc_margin_aurc'], digits=2, bold=(m_key in ab_rc))} \\\\"
        )
    t3.append("\\midrule")
    t3.append("\\multicolumn{8}{@{}l}{\\emph{Unidirectional (\\texttt{v1}, $500$p formal-only edits) vs.\\ bidirectional (\\texttt{v2}) attribution bridge (3 seeds; cf.\\ \\texttt{v2} rows in Table~\\ref{tab:main_id}--\\ref{tab:ood_external})}} \\\\")
    for m_key, m_lbl in br_rows:
        a = agg[m_key]
        t3.append(
            f"{m_lbl} & {pct_ms(a['confirm_wg_E0_E3'])} & {pct_ms(a['id_wg_E0_E3'])} & "
            f"{pct_ms(a['ood_llada_len120_220_auroc'])} & {pct_ms(a['mixset_pure_auroc'])} & "
            f"{pct_ms(a['beemo_auroc'])} & {pct_ms(a['esl_E1_fpr'])} & {pct_ms(a['rc_margin_aurc'], digits=2)} \\\\"
        )
    t3.append("\\midrule")
    t3.append("\\multicolumn{8}{@{}l}{\\emph{Second backbone 5-seed generality:} \\textsc{DeBERTa-v3-large}} \\\\")
    for m_key, m_lbl in db_rows:
        if m_key == "db_v3_cedar_split_op":
            t3.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t3.append(
            f"{m_lbl} & {pct_ms(a['confirm_wg_E0_E3'], bold=(m_key in db_cwg))} & "
            f"{pct_ms(a['id_wg_E0_E3'], bold=(m_key in db_iwg))} & "
            f"{pct_ms(a['ood_llada_len120_220_auroc'], bold=(m_key in db_ll))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in db_ms))} & "
            f"{pct_ms(a['beemo_auroc'], bold=(m_key in db_bm))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in db_esl))} & "
            f"{pct_ms(a['rc_margin_aurc'], digits=2, bold=(m_key in db_rc))} \\\\"
        )
    t3.extend(["\\bottomrule", "\\end{tabular*}%", "}", "\\end{table}"])
    with open(os.path.join(tab_dir, "tab3_ablations_and_backbones.tex"), "w") as f:
        f.write("\n".join(t3) + "\n")

    # =========================================================================
    # TABLE 4 (Appendix): Shortcut Probes, IPW LLaDA & 5,000-Resample Paired Bootstrap + Holm-Bonferroni
    # =========================================================================
    t4 = [
        "\\begin{table}[t]",
        "\\caption{Artifact shortcut probes re-fit on bidirectional \\texttt{train\\_v2}, 4-bin IPW length-reweighted \\textsc{LLaDA-8B} $[120,220]$w AUROC ($n{=}485$, $137$ parents), and $5{,}000$-resample parent/source-family clustered paired bootstrap differences ($\\Delta$ pp [95\\% CI], $p_{\\mathrm{boot}}$; $^{\\ddagger}$Holm--Bonferroni adjusted $p_{\\mathrm{Holm}} < 0.05$ across the 7-test primary confirmatory family).}",
        "\\label{tab:app_probes_sig}",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Probe / Detector} & \\textbf{ID $E_0$} & \\textbf{ID $E_2$} & \\textbf{Confirm AUC} & \\textbf{Conf.\\ Colloq.} & \\textbf{MixSet Pure} & \\textbf{LLaDA Unwt.} & \\textbf{LLaDA IPW} \\\\",
        "\\midrule",
    ]
    for pk, pl in [
        ("probe_length", "Word-Count Logistic Probe"),
        ("probe_surface_12feat", "12-Feature Surface Probe"),
        ("probe_tfidf_e0", "TF-IDF ($1$--$2$g, $E_0$-trained)"),
        ("probe_tfidf_aug", "TF-IDF ($1$--$2$g, $E_{0..3}$-trained)"),
    ]:
        p = pre[pk]
        t4.append(
            f"{pl} & {pct(p['test_id_recoverability']['E0']['auroc'])} & "
            f"{pct(p['test_id_recoverability']['E2']['auroc'])} & "
            f"{pct(p['test_confirm']['overall_auroc'])} & "
            f"{pct(p['test_confirm']['by_register']['colloquial_humanized']['auroc'])} & "
            f"{pct(p['external_mixset']['pure_origin_auroc'])} & "
            f"{pct(p['test_ood_llada']['len120_220_auroc'])} & "
            f"{pct(p['test_ood_llada']['len120_220_ipw_auroc'])} \\\\"
        )
    for ak, al in [
        ("mb_v3_augmented_erm", "Augmented ERM (\\textsc{MB}, 5s)"),
        ("mb_v3_strong_combined_no_nsd", "Strong w/o \\textsc{NSD} (\\textsc{MB}, 5s)"),
        ("mb_v3_cedar_split_op", "\\textbf{\\textsc{CEDAR} Split-Op (\\textsc{MB}, 5s)}"),
        ("db_v3_augmented_erm", "Augmented ERM (\\textsc{DB}, 5s)"),
        ("db_v3_cedar_split_op", "\\textbf{\\textsc{CEDAR} Split-Op (\\textsc{DB}, 5s)}"),
    ]:
        a_m = agg[ak]
        if "cedar" in ak:
            t4.append("\\rowcolor{blue!6}")
        t4.append(
            f"{al} & {pct_ms(a_m['id_recoverability']['E0']['auroc'])} & "
            f"{pct_ms(a_m['id_recoverability']['E2']['auroc'])} & "
            f"{pct_ms(a_m['confirm_auroc'])} & "
            f"{pct_ms(a_m['confirm_by_register']['colloquial_humanized']['auroc'])} & "
            f"{pct_ms(a_m['mixset_pure_auroc'])} & "
            f"{pct_ms(a_m['ood_llada_len120_220_auroc'])} & "
            f"{pct_ms(a_m['ood_llada_len120_220_ipw_auroc'])} \\\\"
        )
    t4.extend(["\\bottomrule", "\\end{tabular*}%", "\\par\\smallskip"])

    def fmt_d(d, holm_key=None):
        if d is None:
            return "--"
        df_s = r1(d["diff"], 1)
        sign = "+" if Decimal(df_s) >= 0 and not df_s.startswith("-") else ""
        lo_s = r1(d["ci95"][0], 1)
        s_lo = "+" if Decimal(lo_s) >= 0 and not lo_s.startswith("-") else ""
        hi_s = r1(d["ci95"][1], 1)
        s_hi = "+" if Decimal(hi_s) >= 0 and not hi_s.startswith("-") else ""
        dag = "$^{\\ddagger}$" if (holm_key and holm.get(holm_key, 1.0) < 0.05) else ""
        return f"{sign}{df_s} [{s_lo}{lo_s},{s_hi}{hi_s}], {r_raw(d['p_boot'], 3)}{dag}"

    t4.extend([
        "\\setlength{\\tabcolsep}{1.05pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}",
        "\\toprule",
        "\\textbf{\\textsc{CEDAR} Split-Op vs.\\ Comparator} & \\textbf{Confirm WG ($160$p)} & \\textbf{ID WG ($200$p)} & \\textbf{MixSet Pure ($120$f)} & \\textbf{LLaDA $[120,220]$w ($137$p)} \\\\",
        "\\midrule",
    ])
    sig_rows = [
        ("mb_v3_cedar_split_op_vs_mb_v3_strong_combined_no_nsd", "vs.\\ Strong w/o \\textsc{NSD} (800p)", "confirm_wg_vs_strong_no_nsd", None, None),
        ("mb_v3_cedar_split_op_vs_mb_v3_strong_combined_fit600_no_nsd", "vs.\\ Strong w/o \\textsc{NSD} (600p)", None, None, None),
        ("mb_v3_cedar_split_op_vs_mb_v3_augmented_erm", "vs.\\ Augmented ERM", "confirm_wg_vs_aug_erm", None, "llada_120_220_vs_aug_erm"),
        ("mb_v3_cedar_split_op_vs_mb_v3_supcon_orth_disentangle", "vs.\\ SupCon + Orth.", "confirm_wg_vs_supcon", None, None),
        ("mb_v3_cedar_split_op_vs_desklib_raid_v1", "vs.\\ \\textsc{Desklib-v1}$^{\\dagger}$", "confirm_wg_vs_desklib", "mixset_pure_vs_desklib", None),
        ("mb_v3_cedar_split_op_vs_mb_v3_ablation_unprotected_nsd", "vs.\\ Unprotected ($\\lambda_{\\mathrm{prot}}{=}0$)", None, None, None),
        ("mb_v3_cedar_split_op_vs_mb_v3_ablation_no_dte", "vs.\\ w/o \\textsc{DTE}", None, None, None),
        ("mb_v3_cedar_split_op_vs_mb_v3_ablation_param_matched_mlp", "vs.\\ Doc-MLP ($122$K)", None, None, None),
        ("db_v3_cedar_split_op_vs_db_v3_augmented_erm", "\\textsc{DB}: vs.\\ Augmented ERM", "deberta_confirm_wg_vs_aug_erm", None, None),
        ("db_v3_cedar_split_op_vs_db_v3_strong_combined_no_nsd", "\\textsc{DB}: vs.\\ Strong w/o \\textsc{NSD}", None, None, None),
    ]
    for sk, cl, hk_cf, hk_ms, hk_ll in sig_rows:
        if sk not in sig:
            continue
        sg = sig[sk]
        t4.append(
            f"{cl} & {fmt_d(sg.get('test_confirm_worst_group_E0_E3'), hk_cf)} & "
            f"{fmt_d(sg.get('test_id_worst_group_E0_E3'))} & "
            f"{fmt_d(sg.get('external_mixset_pure_auroc'), hk_ms)} & "
            f"{fmt_d(sg.get('test_ood_llada_len120_220_auroc'), hk_ll)} \\\\"
        )
    t4.extend(["\\bottomrule", "\\end{tabular*}%", "}", "\\end{table}"])
    with open(os.path.join(tab_dir, "tab4_app_probes_and_bootstrap.tex"), "w") as f:
        f.write("\n".join(t4) + "\n")

    # =========================================================================
    # TABLE 5 (Appendix): Few-Shot K=30 Human-Anchor Split-CRC, Theorem 2 Convergence & Linear Probes
    # =========================================================================
    c_mb = crc["mb_v3_cedar_split_op"]["benchmarks"]
    t5 = [
        "\\begin{table}[t]",
        "\\caption{Top: Zero-shot locked $\\tau_{\\mathrm{cal}}^{\\mathrm{all}}$ vs.\\ $K{=}30$ human-only domain-anchor split-CRC ($200$ cluster-disjoint draws per seed; zero target machine texts; marginal cluster human FPR bounded by $\\alpha_{\\mathrm{fpr}}{=}5\\%$). Middle: Theorem~\\ref{thm:finite_sample} disjoint-pool perturbation ($520$ untouched octets, $N_{\\mathrm{ref}}{=}260$, $200$ trials). Bottom: Seed-42 frozen linear-probe disentanglement (Origin AUC / 3-Way Edit Acc, plus 3-Way Register Acc on \\texttt{test\\_confirm}).}",
        "\\label{tab:app_operational}",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Target Corpus (\\textsc{CEDAR} Split-Op, 5s)} & \\textbf{ZS ClustFPR} & \\textbf{ZS BalAcc} & \\textbf{$K{=}30$ ClustFPR} & \\textbf{$K{=}30$ P90 FPR} & \\textbf{$K{=}30$ MachTPR} & \\textbf{$K{=}30$ BalAcc} \\\\",
        "\\midrule",
    ]
    for bk, bl in [
        ("test_confirm", "Pristine \\texttt{test\\_confirm} ($160$p)"),
        ("external_esl_full", "IELTS ESL Student Essays ($300$f, human-only)"),
        ("external_mixset_pure", "\\textsc{MixSet} Pure Single-Origin ($120$f)"),
        ("external_beemo", "Toloka \\textsc{Beemo} Human/Expert/G4o ($300$p)"),
        ("external_mage_wild", "\\textsc{MAGE} Wild Unseen Domains ($500$h/$500$m)"),
    ]:
        zb = c_mb[bk]["zero_shot_locked_val_calib"]
        fb = c_mb[bk]["few_shot_K30_human_anchor_crc"]
        t5.append(
            f"{bl} & {pct_ms(zb['cluster_human_fpr'])} & {pct_ms(zb['bal_acc'])} & "
            f"{pct_ms(fb['marginal_cluster_human_fpr'], bold=True)} & {pct_ms(fb['p90_cluster_human_fpr_across_draws'])} & "
            f"{pct_ms(fb['machine_tpr'])} & {pct_ms(fb['bal_acc'], bold=True)} \\\\"
        )
    t5.extend(["\\bottomrule", "\\end{tabular*}%", "\\par\\smallskip"])

    t2_v = G["theorem2_empirical_verification"]
    t5.extend([
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Theorem~\\ref{thm:finite_sample} Metric ($N_{\\mathrm{ref}}{=}260$)} & \\textbf{$N{=}16$} & \\textbf{$N{=}32$} & \\textbf{$N{=}64$} & \\textbf{$N{=}128$} & \\textbf{$N{=}256$} & \\textbf{Log-Log Slope} \\\\",
        "\\midrule",
    ])
    bN = t2_v["by_N"]
    slp = t2_v["loglog_exponent_vs_eff_inv_sqrt_n"]
    t5.append(
        "Operator $\\|\\widehat{M}_N - M_{\\mathrm{ref}}\\|_{\\mathrm{op}}$ & "
        + " & ".join(f"{r_raw(bN[str(n)]['op_err_mean'], 3)}$\\pm${r_raw(bN[str(n)]['op_err_sd'], 3)}" for n in [16, 32, 64, 128, 256])
        + f" & {r_raw(slp['operator_error_exponent'], 2)} (th.\\ $1.00$) \\\\"
    )
    t5.append(
        "Shrinker $\\|\\widehat{P}_{\\mathcal{N}} - P_{\\mathcal{N},\\mathrm{ref}}\\|_{\\mathrm{op}}$ & "
        + " & ".join(f"{r_raw(bN[str(n)]['shrinker_err_mean'], 3)}$\\pm${r_raw(bN[str(n)]['shrinker_err_sd'], 3)}" for n in [16, 32, 64, 128, 256])
        + f" & {r_raw(slp['shrinker_error_exponent'], 2)} (th.\\ $1.00$) \\\\"
    )
    t5.append(
        "Confirm P99 $|\\hat{\\ell}(x) - \\ell_{\\mathrm{ref}}(x)|$ & "
        + " & ".join(f"{r_raw(bN[str(n)]['p99_logit_diff_mean'], 3)}" for n in [16, 32, 64, 128, 256])
        + f" & $\\min\\|z\\|_2={r_raw(G['residual_norm_audit']['test_confirm']['min'], 3)}$ \\\\"
    )
    t5.extend(["\\bottomrule", "\\end{tabular*}%", "\\par\\smallskip"])

    lp = G["linear_probe_disentanglement"]
    t5.extend([
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}",
        "\\toprule",
        "\\textbf{Seed-42 Representation} & \\textbf{Confirm (Orig / Edit / Reg)} & \\textbf{ID (Orig / Edit)} & \\textbf{OOD-AR (Orig / Edit)} & \\textbf{LLaDA (Orig / Edit)} \\\\",
        "\\midrule",
    ])
    for rk, rlbl in [
        ("pre_deflation_h_hat", "Unit Trunk $\\hat{h}(x) \\in \\mathbb{S}^{127}$"),
        ("post_deflation_z_orig", "Retained $z_{\\mathrm{orig}}(x) = (I-P_{\\mathcal{N}})\\hat{h}(x)$"),
        ("deflated_nuisance_z_nuis", "Routed Nuisance $z_{\\mathrm{nuis}}(x) = P_{\\mathcal{N}}\\hat{h}(x)$"),
    ]:
        cf = lp["test_confirm"][rk]
        t5.append(
            f"{rlbl} & {pct(cf['origin_auroc'])} / {pct(cf['edit3_acc'])} / {pct(cf['register3_acc'])} & "
            f"{pct(lp['test_id'][rk]['origin_auroc'])} / {pct(lp['test_id'][rk]['edit3_acc'])} & "
            f"{pct(lp['test_ood_ar'][rk]['origin_auroc'])} / {pct(lp['test_ood_ar'][rk]['edit3_acc'])} & "
            f"{pct(lp['test_ood_llada'][rk]['origin_auroc'])} / {pct(lp['test_ood_llada'][rk]['edit3_acc'])} \\\\"
        )
    t5.extend(["\\bottomrule", "\\end{tabular*}%", "}", "\\end{table}"])
    with open(os.path.join(tab_dir, "tab5_app_operational_and_calibration.tex"), "w") as f:
        f.write("\n".join(t5) + "\n")

    print("Generated all 5 V3 LaTeX booktabs tables (zero \\resizebox, native 7pt \\scriptsize!) in", tab_dir)


if __name__ == "__main__":
    generate_tables()
