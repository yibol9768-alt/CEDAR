import os
import gc
import sys
from transformers import AutoTokenizer

sys.path.insert(0, "/content")
from scripts.step4_train_cedar_and_baselines import (
    load_raw_splits,
    compute_dataset_fingerprint,
    pretokenize_split,
    train_and_eval_variant,
)


def main():
    chunk = os.environ.get("W2_CHUNK", "deberta")
    raw_splits = load_raw_splits()
    data_fp = compute_dataset_fingerprint(raw_splits)
    seeds = [42, 1042, 2042]

    if chunk == "deberta":
        db_id = "microsoft/deberta-v3-large"
        print(f"\n>>> [Worker2] Pre-tokenizing all splits for {db_id} (data_fp={data_fp}) ...", flush=True)
        db_tok = AutoTokenizer.from_pretrained(db_id)
        db_splits = {k: pretokenize_split(v, db_tok, max_len=384) for k, v in raw_splits.items()}
        db_variants = [
            ("db_unedited_erm", "unedited_erm"),
            ("db_augmented_erm", "augmented_erm"),
            ("db_supcon_orth_disentangle", "supcon_orth_disentangle"),
            ("db_cedar_full", "cedar_full"),
        ]
        for r_name, mode in db_variants:
            for sd in seeds:
                train_and_eval_variant(r_name, sd, db_id, db_splits, data_fp, mode=mode, epochs=4, lr_backbone=1.5e-5, lr_head=1.5e-4)
        del db_splits, db_tok
        gc.collect()
        return

    mb_id = "answerdotai/ModernBERT-large"
    print(f"\n>>> [Worker2] Pre-tokenizing all splits for {mb_id} (chunk={chunk}) ...", flush=True)
    mb_tok = AutoTokenizer.from_pretrained(mb_id)
    mb_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}

    abl_map = {
        "abl_bottom3": [
            ("mb_ablation_unprotected_nsd", "cedar_unprotected_nsd"),
            ("mb_ablation_no_gdro", "cedar_no_gdro"),
            ("mb_ablation_no_pair", "cedar_no_pair"),
        ],
        "abl_top3": [
            ("mb_ablation_param_matched_mlp", "cedar_param_matched_mlp"),
            ("mb_ablation_no_dte", "cedar_no_dte"),
            ("mb_ablation_no_nsd", "cedar_no_nsd"),
        ],
    }
    for r_name, mode in abl_map[chunk]:
        for sd in seeds:
            train_and_eval_variant(r_name, sd, mb_id, mb_splits, data_fp, mode=mode, epochs=4, lr_backbone=2.5e-5, lr_head=2e-4)


if __name__ == "__main__":
    main()
