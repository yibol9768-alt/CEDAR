import os
import json
from decimal import Decimal, ROUND_HALF_UP


def r1(x, digits=1):
    if x is None:
        return "--"
    q = Decimal("10") ** (-digits)
    val = (Decimal(str(x)) * Decimal("100")).quantize(q, rounding=ROUND_HALF_UP)
    return f"{val:.{digits}f}"


def r_p(x):
    if x is None:
        return "--"
    val = Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{val:.3f}"


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
    res_dir = os.path.join(root, "results")
    tab_dir = os.path.join(root, "paper", "tables")
    os.makedirs(tab_dir, exist_ok=True)

    with open(os.path.join(res_dir, "master_evaluation_summary_v2.json")) as f:
        S = json.load(f)
    with open(os.path.join(res_dir, "nsd_spectrum_and_geometry.json")) as f:
        G = json.load(f)

    pre = S["pretrained_and_probes"]
    agg = S["trained_multi_seed_agg"]
    sig = S["paired_significance_tests"]
    cal_sens = S.get("validation_partition_and_calibration_sensitivity", {})
    sub_aud = S.get("subgroup_and_length_audits", {}).get("models", {})

    mb_rows = [
        ("mb_unedited_erm", "Unedited ERM ($E_0$)$^{\\S}$"),
        ("mb_augmented_erm", "Augmented ERM"),
        ("mb_joint_8class_ce", "Joint 8-Class CE"),
        ("mb_multitask_2head", "Multi-Task 2-Head"),
        ("mb_erm_paired_consist", "ERM + Paired Cons."),
        ("mb_erm_group_dro", "ERM + Group-DRO"),
        ("mb_supcon_orth_disentangle", "SupCon + Orth."),
        ("mb_cedar_full", "\\textbf{\\textsc{CEDAR} (Ours)}"),
    ]
    mb_keys = [k for k, _ in mb_rows]

    # =========================================================================
    # TABLE 1: Controlled Symmetric Recoverability Spectrum (test_id, E0..E4)
    # True unscaled \scriptsize (7pt) via tabular* — zero \resizebox!
    # =========================================================================
    b_f2 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E2"]["fpr_e0cal"], higher_better=False)
    b_f1 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E1"]["fpr_allcal"], higher_better=False)
    b_e0 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E0"]["bal_acc"], higher_better=True)
    b_e1 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E1"]["bal_acc"], higher_better=True)
    b_e2 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E2"]["bal_acc"], higher_better=True)
    b_e3 = best_keys(mb_keys, agg, lambda a: a["id_recoverability"]["E3"]["bal_acc"], higher_better=True)
    b_wg = best_keys(mb_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)

    t1 = []
    t1.append("\\begin{table}[t]")
    t1.append("\\caption{Controlled symmetric recovery on \\texttt{test\\_id} ($2{,}000$ passages, $200$ parents) at locked \\texttt{val\\_calib} thresholds ($\\alpha_{\\mathrm{fpr}}=0.05$; mean$\\pm$SD \\%, \\texttt{ddof=0}, 3 seeds on \\textsc{ModernBERT-large}; best controlled neural row in \\textbf{bold}; $^{\\S}$$1{,}000$ $E_0$ passages cycled to $252$ steps). \\textbf{$E_2$ FPR}$_{\\tau_{E_0}}$ uses $\\tau_{E_0}$; other columns use $\\tau_{\\mathrm{cal}}^{\\mathrm{all}}$.}")
    t1.append("\\label{tab:main_id}")
    t1.append("\\centering")
    t1.append("{\\scriptsize")
    t1.append("\\setlength{\\tabcolsep}{1.8pt}")
    t1.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccc@{}}")
    t1.append("\\toprule")
    t1.append("\\multirow{1}{*}{\\textbf{Detector}} & \\textbf{$E_2$ FPR$_{\\tau_{E_0}}\\downarrow$} & \\textbf{$E_1$ FPR$\\downarrow$} & \\textbf{Bal $E_0\\uparrow$} & \\textbf{Bal $E_1\\uparrow$} & \\textbf{Bal $E_2\\uparrow$} & \\textbf{Bal $E_3\\uparrow$} & \\textbf{Bal $E_4$} & \\textbf{WG ($E_{0..3}$)$\\uparrow$} \\\\")
    t1.append("\\midrule")
    t1.append("\\multicolumn{9}{@{}l}{\\emph{Zero-shot curvature (adapted} \\textsc{Qwen2.5-3B}\\emph{) \\& external-corpus pretrained detectors ($^{\\dagger}$)}} \\\\")

    for p_key, p_lbl in [
        ("fast_detectgpt_qwen3b", "\\textsc{Fast-DetectGPT}"),
        ("binoculars_qwen3b", "\\textsc{Binoculars}"),
        ("radar_vicuna7b", "\\textsc{RADAR-Vicuna-7B}$^{\\dagger}$"),
        ("desklib_raid_v1", "\\textsc{Desklib-v1.01}$^{\\dagger}$"),
    ]:
        r = pre[p_key]["test_id_recoverability"]
        wg = pre[p_key]["test_id_worst_group_acc_E0_E3"]
        t1.append(
            f"{p_lbl} & {pct(r['E2']['fpr_at_tau_e0_5'])} & {pct(r['E1']['fpr_at_tau_all5'])} & "
            f"{pct(r['E0']['bal_acc'])} & {pct(r['E1']['bal_acc'])} & {pct(r['E2']['bal_acc'])} & "
            f"{pct(r['E3']['bal_acc'])} & {pct(r['E4']['bal_acc'])} & {pct(wg)} \\\\"
        )

    t1.append("\\midrule")
    t1.append("\\multicolumn{9}{@{}l}{\\emph{Controlled neural baselines \\&} \\textsc{CEDAR} (\\textsc{ModernBERT-large}\\emph{, 500 training parents, 3 seeds})} \\\\")

    for m_key, m_lbl in mb_rows:
        if m_key == "mb_cedar_full":
            t1.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        rc = a["id_recoverability"]
        t1.append(
            f"{m_lbl} & {pct_ms(rc['E2']['fpr_e0cal'], bold=(m_key in b_f2))} & "
            f"{pct_ms(rc['E1']['fpr_allcal'], bold=(m_key in b_f1))} & "
            f"{pct_ms(rc['E0']['bal_acc'], bold=(m_key in b_e0))} & "
            f"{pct_ms(rc['E1']['bal_acc'], bold=(m_key in b_e1))} & "
            f"{pct_ms(rc['E2']['bal_acc'], bold=(m_key in b_e2))} & "
            f"{pct_ms(rc['E3']['bal_acc'], bold=(m_key in b_e3))} & "
            f"{pct_ms(rc['E4']['bal_acc'])} & "
            f"{pct_ms(a['id_wg_E0_E3'], bold=(m_key in b_wg))} \\\\"
        )

    t1.append("\\bottomrule")
    t1.append("\\end{tabular*}%")
    t1.append("}")
    t1.append("\\end{table}")
    with open(os.path.join(tab_dir, "tab1_main_recoverability.tex"), "w") as f:
        f.write("\n".join(t1) + "\n")

    # =========================================================================
    # TABLE 2: Crossed-OOD (AR + Masked Diffusion) & External Wild Benchmarks
    # True unscaled \scriptsize (7pt) via tabular* — zero \resizebox!
    # =========================================================================
    b_ar_a = best_keys(mb_keys, agg, lambda a: a["ood_ar_auroc"], higher_better=True)
    b_ar_b = best_keys(mb_keys, agg, lambda a: a["ood_ar_mean_bal"], higher_better=True)
    b_ll_a = best_keys(mb_keys, agg, lambda a: a["ood_llada_auroc"], higher_better=True)
    b_ll_b = best_keys(mb_keys, agg, lambda a: a["ood_llada_mean_bal"], higher_better=True)
    b_esl = best_keys(mb_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    b_bm = best_keys(mb_keys, agg, lambda a: a["beemo_auroc"], higher_better=True)
    b_msp = best_keys(mb_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    b_msa = best_keys(mb_keys, agg, lambda a: a["mixset_all_auroc"], higher_better=True)
    b_mg = best_keys(mb_keys, agg, lambda a: a["mage_wild_auroc"], higher_better=True)

    t2 = []
    t2.append("\\begin{table}[t]")
    t2.append("\\caption{Zero-shot transfer to Crossed-OOD splits (unseen \\textsc{Phi-4-mini-instruct} + unseen \\textsc{OLMo-2-7B} AR and \\textsc{LLaDA-8B} diffusion) and four external benchmarks ($^{\\dagger}$external pretraining; $^{\\S}$$1{,}000$ $E_0$ items; other neural rows use $4{,}000$ $E_{0..3}$ items; mean$\\pm$SD, \\texttt{ddof=0}; operational breakdowns in Table~\\ref{tab:app_operational}).}")
    t2.append("\\label{tab:ood_external}")
    t2.append("\\centering")
    t2.append("{\\scriptsize")
    t2.append("\\setlength{\\tabcolsep}{1.1pt}")
    t2.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}")
    t2.append("\\toprule")
    t2.append("\\multirow{2}{*}{\\textbf{Detector}} & \\multicolumn{2}{c}{\\textbf{OOD-AR (\\textsc{OLMo-2})}} & \\multicolumn{2}{c}{\\textbf{OOD-Diff (\\textsc{LLaDA})}} & \\textbf{ESL $E_1$} & \\textbf{Beemo} & \\multicolumn{2}{c}{\\textbf{MixSet AUROC$\\uparrow$}} & \\textbf{MAGE} \\\\")
    t2.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){8-9}")
    t2.append(" & \\textbf{AUC$\\uparrow$} & \\textbf{Bal$\\uparrow$} & \\textbf{AUC$\\uparrow$} & \\textbf{Bal$\\uparrow$} & \\textbf{FPR$\\downarrow$} & \\textbf{AUC$\\uparrow$} & \\textbf{Pure(620)} & \\textbf{All(720)} & \\textbf{AUC$\\uparrow$} \\\\")
    t2.append("\\midrule")

    for p_key, p_lbl in [
        ("fast_detectgpt_qwen3b", "\\textsc{Fast-DetectGPT}"),
        ("binoculars_qwen3b", "\\textsc{Binoculars}"),
        ("radar_vicuna7b", "\\textsc{RADAR-7B}$^{\\dagger}$"),
        ("desklib_raid_v1", "\\textsc{Desklib-v1}$^{\\dagger}$"),
    ]:
        p = pre[p_key]
        t2.append(
            f"{p_lbl} & {pct(p['test_ood_ar']['overall_auroc'])} & {pct(p['test_ood_ar']['mean_bal_acc'])} & "
            f"{pct(p['test_ood_llada']['overall_auroc'])} & {pct(p['test_ood_llada']['mean_bal_acc'])} & "
            f"{pct(p['external_esl_fairness']['esl_E1_polished_fpr_at_tau_all5'])} & "
            f"{pct(p['external_beemo']['overall_auroc'])} & "
            f"{pct(p['external_mixset']['pure_origin_auroc'])} & {pct(p['external_mixset']['all_720_auroc'])} & "
            f"{pct(p['external_mage_wild']['auroc'])} \\\\"
        )

    t2.append("\\midrule")
    for m_key, m_lbl in mb_rows:
        if m_key == "mb_cedar_full":
            t2.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t2.append(
            f"{m_lbl} & {pct_ms(a['ood_ar_auroc'], bold=(m_key in b_ar_a))} & "
            f"{pct_ms(a['ood_ar_mean_bal'], bold=(m_key in b_ar_b))} & "
            f"{pct_ms(a['ood_llada_auroc'], bold=(m_key in b_ll_a))} & "
            f"{pct_ms(a['ood_llada_mean_bal'], bold=(m_key in b_ll_b))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in b_esl))} & "
            f"{pct_ms(a['beemo_auroc'], bold=(m_key in b_bm))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in b_msp))} & "
            f"{pct_ms(a['mixset_all_auroc'], bold=(m_key in b_msa))} & "
            f"{pct_ms(a['mage_wild_auroc'], bold=(m_key in b_mg))} \\\\"
        )

    t2.append("\\bottomrule")
    t2.append("\\end{tabular*}%")
    t2.append("}")
    t2.append("\\end{table}")
    with open(os.path.join(tab_dir, "tab2_ood_and_external.tex"), "w") as f:
        f.write("\n".join(t2) + "\n")

    # =========================================================================
    # TABLE 3: Leave-One-Out Ablations, Parameter-Matched Control & Second Backbone
    # True unscaled \scriptsize (7pt) via tabular* — zero \resizebox!
    # =========================================================================
    ab_rows = [
        ("mb_cedar_full", "\\textbf{\\textsc{CEDAR} Full ($r{=}32, \\lambda_{\\mathrm{prot}}{=}0.5$)}"),
        ("mb_ablation_no_nsd", "\\quad w/o \\textsc{NSD} ($r=0$)"),
        ("mb_ablation_unprotected_nsd", "\\quad w/ Unprotected \\textsc{NSD} ($\\lambda_{\\mathrm{prot}}=0$)"),
        ("mb_ablation_no_dte", "\\quad w/o Kinematics (\\textsc{DTE})"),
        ("mb_ablation_param_matched_mlp", "\\quad w/ Doc-MLP Control ($122$K)"),
        ("mb_ablation_no_pair", "\\quad w/o Paired Alignment ($\\lambda_{\\mathrm{pair}}=0$)"),
        ("mb_ablation_no_gdro", "\\quad w/o Cond.\\ Group-DRO ($\\eta_{\\mathrm{dro}}=0$)"),
    ]
    ab_keys = [k for k, _ in ab_rows if k in agg]

    ab_wg = best_keys(ab_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)
    ab_ar = best_keys(ab_keys, agg, lambda a: a["ood_ar_mean_bal"], higher_better=True)
    ab_ll = best_keys(ab_keys, agg, lambda a: a["ood_llada_auroc"], higher_better=True)
    ab_esl = best_keys(ab_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    ab_ms = best_keys(ab_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    ab_rcm = best_keys(ab_keys, agg, lambda a: a["rc_margin_aurc"], higher_better=False)
    ab_rci = best_keys(ab_keys, agg, lambda a: a["rc_intens_aurc"], higher_better=False)

    db_rows = [
        ("db_unedited_erm", "\\textsc{DeBERTa-v3}: Unedited ERM$^{\\S}$"),
        ("db_augmented_erm", "\\textsc{DeBERTa-v3}: Augmented ERM"),
        ("db_supcon_orth_disentangle", "\\textsc{DeBERTa-v3}: SupCon + Orth."),
        ("db_cedar_full", "\\textbf{\\textsc{DeBERTa-v3}: \\textsc{CEDAR} (Ours)}"),
    ]
    db_keys = [k for k, _ in db_rows if k in agg]
    db_wg = best_keys(db_keys, agg, lambda a: a["id_wg_E0_E3"], higher_better=True)
    db_ar = best_keys(db_keys, agg, lambda a: a["ood_ar_mean_bal"], higher_better=True)
    db_ll = best_keys(db_keys, agg, lambda a: a["ood_llada_auroc"], higher_better=True)
    db_esl = best_keys(db_keys, agg, lambda a: a["esl_E1_fpr"], higher_better=False)
    db_ms = best_keys(db_keys, agg, lambda a: a["mixset_pure_auroc"], higher_better=True)
    db_rcm = best_keys(db_keys, agg, lambda a: a["rc_margin_aurc"], higher_better=False)
    db_rci = best_keys(db_keys, agg, lambda a: a["rc_intens_aurc"], higher_better=False)

    t3 = []
    t3.append("\\begin{table}[t]")
    t3.append("\\caption{Ablations, parameter-matched MLP ($122{,}296$ vs.\\ $122{,}255$ params excl.\\ shared fusion), signal protection ($\\lambda_{\\mathrm{prot}}=0$), and \\textsc{DeBERTa-v3-large} verification (mean$\\pm$SD, \\texttt{ddof=0}, 3 seeds). $\\mathrm{AURC}_{m}, \\mathrm{AURC}_{c}$ report normalized selective error (\\%) over $20\\%$--$100\\%$ coverage on \\texttt{test\\_id} ($E_{0..4}$).}")
    t3.append("\\label{tab:ablations}")
    t3.append("\\centering")
    t3.append("{\\scriptsize")
    t3.append("\\setlength{\\tabcolsep}{2.2pt}")
    t3.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}")
    t3.append("\\toprule")
    t3.append("\\textbf{Variant / Backbone} & \\textbf{ID WG$\\uparrow$} & \\textbf{OOD-AR$\\uparrow$} & \\textbf{LLaDA$\\uparrow$} & \\textbf{ESL FPR$\\downarrow$} & \\textbf{MixSet(620)$\\uparrow$} & \\textbf{AURC$_{m}\\downarrow$} & \\textbf{AURC$_{c}\\downarrow$} \\\\")
    t3.append("\\midrule")
    t3.append("\\multicolumn{8}{@{}l}{\\textsc{ModernBERT-large} \\emph{leave-one-out \\& mechanistic controls (3 seeds)}} \\\\")

    for m_key, m_lbl in ab_rows:
        if m_key not in agg:
            continue
        if m_key == "mb_cedar_full":
            t3.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t3.append(
            f"{m_lbl} & {pct_ms(a['id_wg_E0_E3'], bold=(m_key in ab_wg))} & "
            f"{pct_ms(a['ood_ar_mean_bal'], bold=(m_key in ab_ar))} & "
            f"{pct_ms(a['ood_llada_auroc'], bold=(m_key in ab_ll))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in ab_esl))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in ab_ms))} & "
            f"{pct_ms(a['rc_margin_aurc'], digits=2, bold=(m_key in ab_rcm))} & "
            f"{pct_ms(a['rc_intens_aurc'], digits=2, bold=(m_key in ab_rci))} \\\\"
        )

    t3.append("\\midrule")
    t3.append("\\multicolumn{8}{@{}l}{\\emph{Second backbone verification:} \\textsc{DeBERTa-v3-large} \\emph{(3 seeds)}} \\\\")
    for m_key, m_lbl in db_rows:
        if m_key not in agg:
            continue
        if m_key == "db_cedar_full":
            t3.append("\\rowcolor{blue!6}")
        a = agg[m_key]
        t3.append(
            f"{m_lbl} & {pct_ms(a['id_wg_E0_E3'], bold=(m_key in db_wg))} & "
            f"{pct_ms(a['ood_ar_mean_bal'], bold=(m_key in db_ar))} & "
            f"{pct_ms(a['ood_llada_auroc'], bold=(m_key in db_ll))} & "
            f"{pct_ms(a['esl_E1_fpr'], bold=(m_key in db_esl))} & "
            f"{pct_ms(a['mixset_pure_auroc'], bold=(m_key in db_ms))} & "
            f"{pct_ms(a['rc_margin_aurc'], digits=2, bold=(m_key in db_rcm))} & "
            f"{pct_ms(a['rc_intens_aurc'], digits=2, bold=(m_key in db_rci))} \\\\"
        )

    t3.append("\\bottomrule")
    t3.append("\\end{tabular*}%")
    t3.append("}")
    t3.append("\\end{table}")
    with open(os.path.join(tab_dir, "tab3_ablations_and_backbones.tex"), "w") as f:
        f.write("\n".join(t3) + "\n")

    # =========================================================================
    # TABLE 4 (Appendix): Two Unscaled \scriptsize Subtables (Zero \resizebox!)
    #   Part A: Shortcut Probes & Length-Matched / IPW LLaDA AUROC
    #   Part B: Clustered Paired Bootstrap Differences (with exact 137 contributing parents!)
    # =========================================================================
    t4 = []
    t4.append("\\begin{table}[t]")
    t4.append("\\caption{Artifact shortcut probes, common-range ($[120,220]$w, $n=485$ across $137$ contributing parents of the $150$-parent split) and 4-bin IPW length-reweighted \\textsc{LLaDA-8B} AUROC, and 1,000-resample parent/source-family clustered paired bootstrap differences ($\\Delta$ in percentage points [95\\% percentile CI], two-sided bootstrap sign-tail proportion $p_{\\mathrm{boot}}$ conditional on saved 3-seed runs; \\textsc{MixSet} clustered by 120 original-source families).}")
    t4.append("\\label{tab:app_probes_sig}")
    t4.append("\\centering")
    t4.append("{\\scriptsize")
    t4.append("\\setlength{\\tabcolsep}{2.2pt}")
    t4.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}")
    t4.append("\\toprule")
    t4.append("\\multirow{2}{*}{\\textbf{Probe / Detector}} & \\multicolumn{3}{c}{\\textbf{ID \\texttt{test\\_id} AUROC (\\%)}} & \\multicolumn{2}{c}{\\textbf{ID Length ($E_{0..2}$)}} & \\multicolumn{2}{c}{\\textbf{LLaDA $[120,220]$w}} \\\\")
    t4.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")
    t4.append(" & \\textbf{$E_0$} & \\textbf{$E_2$} & \\textbf{$E_4$} & \\textbf{$\\le 180$w} & \\textbf{$>180$w} & \\textbf{Unwt.} & \\textbf{IPW} \\\\")
    t4.append("\\midrule")
    for pk, pl in [
        ("probe_length", "Word-Count Logistic Probe"),
        ("probe_surface_12feat", "12-Feature Surface Probe"),
        ("probe_tfidf_e0", "TF-IDF ($1$--$2$g, $E_0$-trained)"),
        ("probe_tfidf_aug", "TF-IDF ($1$--$2$g, $E_{0..3}$-trained)"),
    ]:
        p = pre[pk]
        t4.append(
            f"{pl} & {pct(p['test_id_recoverability']['E0']['auroc'])} & {pct(p['test_id_recoverability']['E2']['auroc'])} & "
            f"{pct(p['test_id_recoverability']['E4']['auroc'])} & "
            f"{pct(p['test_id_length_stratified_E0_E2']['short_le180_auroc'])} & "
            f"{pct(p['test_id_length_stratified_E0_E2']['long_gt180_auroc'])} & -- & -- \\\\"
        )
    for ak, al in [
        ("mb_unedited_erm", "Unedited ERM (\\textsc{ModernBERT})"),
        ("mb_augmented_erm", "Augmented ERM (\\textsc{ModernBERT})"),
        ("mb_erm_paired_consist", "Paired Consist.\\ (\\textsc{ModernBERT})"),
        ("mb_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{ModernBERT})}"),
        ("db_augmented_erm", "Augmented ERM (\\textsc{DeBERTa-v3})"),
        ("db_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{DeBERTa-v3})}"),
    ]:
        a_m = agg[ak]
        sa = sub_aud.get(ak, {})
        if ak in {"mb_cedar_full", "db_cedar_full"}:
            t4.append("\\rowcolor{blue!6}")
        t4.append(
            f"{al} & {pct_ms(a_m['id_recoverability']['E0']['auroc'])} & "
            f"{pct_ms(a_m['id_recoverability']['E2']['auroc'])} & "
            f"{pct_ms(a_m['id_recoverability']['E4']['auroc'])} & "
            f"{pct_ms(a_m['id_short_le180_auroc'])} & {pct_ms(a_m['id_long_gt180_auroc'])} & "
            f"{pct_ms(sa.get('llada_len120_220_auroc'))} & {pct_ms(sa.get('llada_len120_220_ipw_auroc'))} \\\\"
        )
    t4.append("\\bottomrule")
    t4.append("\\end{tabular*}%")
    t4.append("\\par\\medskip")

    def fmt_d(d):
        if d is None:
            return "--"
        df_s = r1(d["diff"], 1)
        sign = "+" if Decimal(df_s) >= 0 and not df_s.startswith("-") else ""
        lo_s = r1(d["ci95"][0], 1)
        s_lo = "+" if Decimal(lo_s) >= 0 and not lo_s.startswith("-") else ""
        hi_s = r1(d["ci95"][1], 1)
        s_hi = "+" if Decimal(hi_s) >= 0 and not hi_s.startswith("-") else ""
        return f"{sign}{df_s} [{s_lo}{lo_s},{s_hi}{hi_s}], {r_p(d['p_boot'])}"

    t4.append("\\setlength{\\tabcolsep}{1.2pt}")
    t4.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}")
    t4.append("\\toprule")
    t4.append("\\textbf{\\textsc{CEDAR} vs.\\ Comparator} & \\textbf{ID WG ($E_{0..3}$, 200p)} & \\textbf{MixSet Pure (620, 120f)} & \\textbf{MixSet All (720, 120f)} & \\textbf{LLaDA $[120,220]$w AUC} \\\\")
    t4.append("\\emph{($\\Delta$ pp [95\\% CI], $p_{\\mathrm{boot}}$)} & \\emph{8-cell min acc} & \\emph{350 hum + 270 mach} & \\emph{450 hum + 270 mach} & \\emph{137p ($n=485$, of 150p)} \\\\")
    t4.append("\\midrule")

    for sk, cl in [
        ("mb_cedar_full_vs_mb_unedited_erm", "vs.\\ Unedited ERM"),
        ("mb_cedar_full_vs_mb_augmented_erm", "vs.\\ Augmented ERM"),
        ("mb_cedar_full_vs_mb_supcon_orth_disentangle", "vs.\\ SupCon + Orth."),
        ("mb_cedar_full_vs_mb_erm_paired_consist", "vs.\\ Paired Consist."),
        ("mb_cedar_full_vs_mb_ablation_no_nsd", "vs.\\ w/o \\textsc{NSD} ($r=0$)"),
        ("mb_cedar_full_vs_mb_ablation_no_dte", "vs.\\ w/o \\textsc{DTE}"),
        ("mb_cedar_full_vs_mb_ablation_param_matched_mlp", "vs.\\ Doc-MLP ($122$K)"),
        ("mb_cedar_full_vs_mb_ablation_no_pair", "vs.\\ w/o Alignment"),
        ("mb_cedar_full_vs_mb_ablation_unprotected_nsd", "vs.\\ Unprot.\\ ($\\lambda_{\\mathrm{prot}}{=}0$)"),
        ("db_cedar_full_vs_db_augmented_erm", "\\textsc{DB}: vs.\\ Aug.\\ ERM"),
    ]:
        if sk not in sig:
            continue
        sg = sig[sk]
        t4.append(
            f"{cl} & {fmt_d(sg.get('test_id_worst_group_E0_E3'))} & "
            f"{fmt_d(sg.get('external_mixset_pure_auroc'))} & "
            f"{fmt_d(sg.get('external_mixset_all_auroc'))} & "
            f"{fmt_d(sg.get('test_ood_llada_len120_220_auroc'))} \\\\"
        )
    t4.append("\\bottomrule")
    t4.append("\\end{tabular*}%")
    t4.append("}")
    t4.append("\\end{table}")
    with open(os.path.join(tab_dir, "tab4_app_probes_and_bootstrap.tex"), "w") as f:
        f.write("\n".join(t4) + "\n")

    # =========================================================================
    # TABLE 5 (Appendix): Three Unscaled \scriptsize / \footnotesize Subtables
    # Zero \resizebox — true 7.0pt to 8.0pt print text across all 3 panels!
    # =========================================================================
    t5 = []
    t5.append("\\begin{table}[t]")
    t5.append("\\caption{Operational locked-threshold ($\\tau_{\\mathrm{cal}}^{\\mathrm{all}}$) transfer across OOD/external subsets (top; \\textsc{MixSet} subsets: polished human $n=200$, humanized MGT $n=100$; \\textsc{Beemo}: $n=300$ per subset; \\textsc{MAGE Wild}: $500$ human / $500$ machine), selection-contaminated post-hoc source-stratified validation recalibration sensitivity ($30$ \\textsc{MAGE} + $30$ \\textsc{No Robots} parents; middle), and single-seed (Seed-42) frozen linear-probe diagnostics (bottom). Top and middle panels report 3-seed mean$\\pm$SD (\\texttt{ddof=0}) for neural models.}")
    t5.append("\\label{tab:app_operational}")
    t5.append("\\centering")
    t5.append("{\\scriptsize")
    t5.append("\\setlength{\\tabcolsep}{1.2pt}")
    t5.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}")
    t5.append("\\toprule")
    t5.append("\\multirow{2}{*}{\\textbf{Detector}} & \\textbf{LLaDA} & \\multicolumn{2}{c}{\\textbf{MixSet}} & \\multicolumn{4}{c}{\\textbf{Beemo ($n=300$ / subset)}} & \\multicolumn{2}{c}{\\textbf{MAGE ($500$/$500$)}} \\\\")
    t5.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-8} \\cmidrule(lr){9-10}")
    t5.append(" & \\textbf{WG} & \\textbf{PolFPR} & \\textbf{HumTPR} & \\textbf{HumFPR} & \\textbf{RawTPR} & \\textbf{G4oTPR} & \\textbf{ExpTPR} & \\textbf{HumFPR} & \\textbf{MachTPR} \\\\")
    t5.append("\\midrule")
    for pk, pl in [
        ("fast_detectgpt_qwen3b", "\\textsc{Fast-DetectGPT}"),
        ("binoculars_qwen3b", "\\textsc{Binoculars}"),
        ("radar_vicuna7b", "\\textsc{RADAR-7B}$^{\\dagger}$"),
        ("desklib_raid_v1", "\\textsc{Desklib-v1}$^{\\dagger}$"),
    ]:
        p = pre[pk]
        t5.append(
            f"{pl} & {pct(p['test_ood_llada']['worst_group_acc'])} & "
            f"{pct(p['external_mixset']['hwt_llm_polished_fpr_tau_all5'])} & "
            f"{pct(p['external_mixset']['mgt_humanized_tpr_tau_all5'])} & "
            f"{pct(p['external_beemo']['human_fpr_tau_all5'])} & "
            f"{pct(p['external_beemo']['mgt_raw_tpr_tau_all5'])} & "
            f"{pct(p['external_beemo']['mgt_gpt4o_edited_tpr_tau_all5'])} & "
            f"{pct(p['external_beemo']['mgt_expert_edited_tpr_tau_all5'])} & "
            f"{pct(p['external_mage_wild']['human_fpr_tau_all5'])} & "
            f"{pct(p['external_mage_wild']['machine_tpr_tau_all5'])} \\\\"
        )
    for mk, ml in [
        ("mb_unedited_erm", "Unedited ERM (\\textsc{MB})"),
        ("mb_augmented_erm", "Augmented ERM (\\textsc{MB})"),
        ("mb_erm_paired_consist", "Paired Cons.\\ (\\textsc{MB})"),
        ("mb_supcon_orth_disentangle", "SupCon + Orth (\\textsc{MB})"),
        ("mb_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{MB})}"),
        ("db_augmented_erm", "Augmented ERM (\\textsc{DB})"),
        ("db_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{DB})}"),
    ]:
        a = agg[mk]
        if mk in {"mb_cedar_full", "db_cedar_full"}:
            t5.append("\\rowcolor{blue!6}")
        t5.append(
            f"{ml} & {pct_ms(a['ood_llada_wg_acc'])} & "
            f"{pct_ms(a['mixset_hwt_pol_fpr'])} & {pct_ms(a['mixset_mgt_hum_tpr'])} & "
            f"{pct_ms(a['beemo_hum_fpr'])} & {pct_ms(a['beemo_raw_tpr'])} & "
            f"{pct_ms(a['beemo_gpt4o_tpr'])} & {pct_ms(a['beemo_exp_tpr'])} & "
            f"{pct_ms(a['mage_wild_hum_fpr'])} & {pct_ms(a['mage_wild_mach_tpr'])} \\\\"
        )
    t5.append("\\bottomrule")
    t5.append("\\end{tabular*}%")

    if cal_sens:
        t5.append("\\par\\smallskip")
        t5.append("\\setlength{\\tabcolsep}{2.4pt}")
        t5.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}")
        t5.append("\\toprule")
        t5.append("\\multirow{2}{*}{\\textbf{Detector (\\texttt{test\\_id} $E_{0..3}$)}} & \\textbf{Locked NR$\\to$NR} & \\textbf{Locked NR$\\to$MAGE} & \\textbf{Strat.\\ Cal.$^*$} & \\textbf{Strat.\\ Cal.$^*$} & \\textbf{Locked} \\\\")
        t5.append(" & \\textbf{Hum FPR} & \\textbf{Hum FPR} & \\textbf{Hum FPR} & \\textbf{ID WG} & \\textbf{ID WG} \\\\")
        t5.append("\\midrule")
        for mk, ml in [
            ("mb_unedited_erm", "Unedited ERM (\\textsc{ModernBERT})"),
            ("mb_augmented_erm", "Augmented ERM (\\textsc{ModernBERT})"),
            ("mb_supcon_orth_disentangle", "SupCon + Orth.\\ (\\textsc{ModernBERT})"),
            ("mb_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{ModernBERT})}"),
            ("db_augmented_erm", "Augmented ERM (\\textsc{DeBERTa-v3})"),
            ("db_cedar_full", "\\textbf{\\textsc{CEDAR} (\\textsc{DeBERTa-v3})}"),
        ]:
            if mk not in cal_sens:
                continue
            cs = cal_sens[mk]
            if mk in {"mb_cedar_full", "db_cedar_full"}:
                t5.append("\\rowcolor{blue!6}")
            t5.append(
                f"{ml} & {pct_ms(cs['locked_nr_to_nr_human_fpr_E0_E3'])} & "
                f"{pct_ms(cs['locked_nr_to_mg_human_fpr_E0_E3'])} & "
                f"{pct_ms(cs['posthoc_stratified_30mg_30nr_human_fpr_E0_E3'])} & "
                f"{pct_ms(cs['posthoc_stratified_30mg_30nr_id_wg_E0_E3'])} & "
                f"{pct_ms(agg[mk]['id_wg_E0_E3'])} \\\\"
            )
        t5.append("\\bottomrule")
        t5.append("\\end{tabular*}%")

    lp = G.get("linear_probe_disentanglement", {})
    if lp:
        t5.append("\\par\\smallskip")
        t5.append("\\setlength{\\tabcolsep}{3.0pt}")
        t5.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccc@{}}")
        t5.append("\\toprule")
        t5.append("\\multirow{2}{*}{\\textbf{Seed-42 \\textsc{CEDAR} Representation}} & \\textbf{ID (\\texttt{test\\_id})} & \\textbf{OOD-AR (\\texttt{test\\_ood\\_ar})} & \\textbf{LLaDA-8B (\\texttt{test\\_ood\\_llada})} \\\\")
        t5.append(" & \\emph{Orig.\\ AUC / 3-Way Edit Acc} & \\emph{Orig.\\ AUC / 3-Way Edit Acc} & \\emph{Orig.\\ AUC / 3-Way Edit Acc} \\\\")
        t5.append("\\midrule")
        for rk, rlbl in [
            ("pre_deflation_h", "Pre-Deflation Trunk $h(x)$"),
            ("post_deflation_z_orig", "Retained Representation $z_{\\mathrm{orig}}(x)$"),
            ("deflated_nuisance_z_nuis", "Attenuated Component $z_{\\mathrm{nuis}}(x)$"),
        ]:
            t5.append(
                f"{rlbl} & {pct(lp['test_id'][rk]['origin_auroc'])} / {pct(lp['test_id'][rk]['edit3_acc'])} & "
                f"{pct(lp['test_ood_ar'][rk]['origin_auroc'])} / {pct(lp['test_ood_ar'][rk]['edit3_acc'])} & "
                f"{pct(lp['test_ood_llada'][rk]['origin_auroc'])} / {pct(lp['test_ood_llada'][rk]['edit3_acc'])} \\\\"
            )
        t5.append("\\bottomrule")
        t5.append("\\end{tabular*}%")

    t5.append("}")
    t5.append("\\end{table}")
    with open(os.path.join(tab_dir, "tab5_app_operational_and_calibration.tex"), "w") as f:
        f.write("\n".join(t5) + "\n")

    print("Generated all 5 LaTeX booktabs tables (zero \\resizebox, true 7pt \\scriptsize!) in", tab_dir)


if __name__ == "__main__":
    generate_tables()
