import os
import re
import gc
import sys
import time
import json
import random
import hashlib
import platform
import subprocess
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

V1_DIR = "/content/cedar_data"
V2_DIR = "/content/cedar_data_v2"
os.makedirs(V2_DIR, exist_ok=True)

SEED = 20261114
SYS_PLAIN = (
    "You are an expert writer and editor. Follow the user's instruction faithfully and output ONLY "
    "the requested passage text itself, with no introductory remarks, titles, meta-commentary, or markdown fences."
)

PREAMBLE_RE = re.compile(
    r"^(?:sure[,!.]?|certainly[,!.]?|here\s+is\s+[^:\n]{0,60}:|revised\s+(?:passage|text|version)\s*:|"
    r"edited\s+(?:passage|text|version)\s*:|paraphrased\s+(?:passage|text|version)\s*:|"
    r"rewritten\s+(?:passage|text|version)\s*:|summary\s*:)\s*",
    re.IGNORECASE,
)


def atomic_save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def sanitize_output(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t).strip()
    for _ in range(3):
        t_new = PREAMBLE_RE.sub("", t).strip()
        if t_new != t:
            t = t_new
        else:
            break
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "“" and t[-1] == "”")):
        t = t[1:-1].strip()
    return clean_text(t)


def word_count(t: str) -> int:
    return len(t.split())


def norm_prefix_hash(t: str) -> str:
    norm = re.sub(r"\W+", "", t.lower())[:300]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def exact_sha256(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def det_rank_key(tag: str, item_id: str) -> str:
    return hashlib.sha256(f"{tag}|{item_id}".encode("utf-8")).hexdigest()


def token_jaccard_dist(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


# =====================================================================
# Decoupled Full-Factorial Prompt Pools (Bidirectional Formal + Humanized)
# =====================================================================
DRAFT_TEMPLATES_SEEN_V2 = [
    "{prompt}\n\nWrite a natural, coherent passage of roughly {target_wc} words. Output only the passage.",
    "Respond directly to the following writing prompt in approximately {target_wc} words, outputting only the text itself:\n\n{prompt}",
    "Develop a clear, substantive {target_wc}-word passage addressing the prompt below. Return only the body text:\n\n{prompt}",
    "Write a candid, natural-sounding human-style passage of about {target_wc} words in response to the prompt below, varying sentence rhythm naturally. Output only the passage:\n\n{prompt}",
]

# Each edit level has 4 templates: indices 0,1 = Formalizing/Polishing (reg=1); indices 2,3 = Conversational/Humanizing (reg=2)
EDIT_TEMPLATES_SEEN_V2 = {
    "E1": [
        # Formal (reg=1)
        (1, "Lightly copyedit the following passage for grammar, spelling, punctuation, and smooth formal fluency. Keep the exact sentence order, factual claims, examples, and structure intact. Output only the copyedited passage:\n\n{text}"),
        (1, "Proofread and lightly polish grammar and awkward phrasing in the text below without changing its sentence sequence, arguments, or specific details. Return only the polished text:\n\n{text}"),
        # Humanized / Conversational (reg=2)
        (2, "Lightly loosen stiff phrasing in the passage below so it reads with natural everyday contractions and relaxed human punctuation, while strictly keeping the exact sentence order and every factual claim intact. Output only the lightly revised passage:\n\n{text}"),
        (2, "Give the text below a light human conversational touch—using natural contractions and plain wording where phrasing feels overly rigid—without reordering any sentences or altering any facts or arguments. Return only the revised text:\n\n{text}"),
    ],
    "E2": [
        # Formal (reg=1)
        (1, "Paraphrase the following passage at the sentence level using fresh, articulate vocabulary and varied formal clause structures, while keeping all underlying points, entities, and paragraph progression intact. Output only the paraphrased passage:\n\n{text}"),
        (1, "Rewrite each sentence of the passage below with substantially different formal wording and syntax while preserving every factual claim and the overall sequence of ideas. Return only the rewritten text:\n\n{text}"),
        # Humanized / Conversational (reg=2)
        (2, "Rewrite each sentence of the passage below in a candid, colloquial, naturally human voice—mixing short direct sentences with longer ones, using everyday idioms and contractions—while keeping every factual claim and the sequence of ideas intact. Output only the rewritten passage:\n\n{text}"),
        (2, "Paraphrase the text below sentence by sentence so it sounds like an unvarnished, spontaneous human writer explaining the exact same points in natural conversational rhythm, without adding or dropping any substantive facts. Return only the rewritten text:\n\n{text}"),
    ],
    "E3": [
        # Formal (reg=1)
        (1, "Substantially reorganize and rewrite the passage below in polished expository prose: reorder points where natural, merge or split sentences, and reshape transitions while retaining the core thesis and key facts. Output only the restructured passage:\n\n{text}"),
        (1, "Restructure the discourse organization of the following text by rearranging sentence order, combining or dividing clauses, and recasting formal transitions while keeping the central content accurate. Return only the rewritten passage:\n\n{text}"),
        # Humanized / Conversational (reg=2)
        (2, "Substantially reorganize and rewrite the passage below as an engaging, natural human explanation: reorder points freely, alternate punchy remarks with longer reflective sentences, and use conversational transitions and contractions while keeping all core facts accurate. Output only the rewritten passage:\n\n{text}"),
        (2, "Reshape and reorder the discourse of the text below into a candid, conversational human narrative—varying sentence cadence, splitting stiff blocks, and using natural spoken flow—while faithfully retaining the thesis and key facts. Return only the rewritten passage:\n\n{text}"),
    ],
}

# Strictly disjoint unseen templates for Pristine Confirmation Benchmark (`split_test_confirm.json`)
DRAFT_TEMPLATES_CONFIRM = [
    "Draft a self-contained response of roughly {target_wc} words to the topic below. Provide solely the passage text without titles or prefaces:\n\n{prompt}",
    "Write an authentic, engaging {target_wc}-word passage on the following prompt, outputting nothing except the passage itself:\n\n{prompt}",
]

EDIT_TEMPLATES_CONFIRM = {
    "E1": [
        (1, "Perform a restrained editorial pass on the passage below to tidy mechanics and phrasing stiffness while leaving sentence order and all claims untouched. Output solely the edited text:\n\n{text}"),
        (2, "Slightly relax overly formal wording in the passage below using natural contractions and plain phrasing while leaving sentence order and every factual detail unchanged. Output solely the revised text:\n\n{text}"),
    ],
    "E2": [
        (1, "Recast every sentence of the passage below in fresh, well-crafted prose with altered syntax and vocabulary while preserving all propositions and their order. Output solely the recast passage:\n\n{text}"),
        (2, "Reword every sentence of the passage below into informal, candid everyday English with uneven natural sentence lengths while preserving all factual propositions and their order. Output solely the reworded passage:\n\n{text}"),
    ],
    "E3": [
        (1, "Reorganize the paragraph structure and sentence progression of the passage below in crisp analytical prose while preserving its core argument and facts. Output solely the reorganized passage:\n\n{text}"),
        (2, "Reorganize and retell the passage below in a lively, colloquial first-hand explanatory voice—reshuffling sentence order and varying rhythm—while preserving its core argument and facts. Output solely the rewritten passage:\n\n{text}"),
    ],
}


def resolve_hf_provenance():
    api = HfApi()
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/Falcon3-7B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "GSAI-ML/LLaDA-8B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "ibm-granite/granite-3.1-8b-instruct",
        "answerdotai/ModernBERT-large",
        "microsoft/deberta-v3-large",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct",
        "TrustSafeAI/RADAR-Vicuna-7B",
        "desklib/ai-text-detector-v1.01",
    ]
    datasets = [
        "HuggingFaceH4/no_robots",
        "yaful/MAGE",
        "liamdugan/raid",
        "chillies/IELTS-writing-task-2-evaluation",
        "ONE-Lab/MixSet",
        "toloka/beemo",
    ]
    model_shas = {}
    for m in models:
        info = api.model_info(m)
        model_shas[m] = info.sha
    dataset_shas = {}
    for d in datasets:
        info = api.dataset_info(d)
        dataset_shas[d] = info.sha

    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).strip().splitlines()
    with open(os.path.join(V2_DIR, "requirements_lock.txt"), "w") as f:
        f.write("\n".join(pip_freeze) + "\n")

    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "device_name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }

    prov = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "gpu_info": gpu_info,
        "hf_model_commit_shas": model_shas,
        "hf_dataset_commit_shas": dataset_shas,
        "pip_package_count": len(pip_freeze),
    }
    atomic_save_json(os.path.join(V2_DIR, "provenance_lock.json"), prov)
    print(f"[PROVENANCE LOCKED] {len(model_shas)} HF models, {len(dataset_shas)} HF datasets, {len(pip_freeze)} pip packages.")
    return prov


def build_curated_manifests():
    t0 = time.time()
    prov_path = os.path.join(V2_DIR, "provenance_lock.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else resolve_hf_provenance()

    parents_v1 = json.load(open(os.path.join(V1_DIR, "raw_parents.json")))
    split_val = json.load(open(os.path.join(V1_DIR, "split_val.json")))

    # 1. Deterministic source-stratified validation manifest (30 MAGE + 30 No Robots in each of val_select and val_calib)
    val_by_dom = defaultdict(set)
    for r in split_val:
        val_by_dom[r["domain"]].add(r["parent_id"])
    assert len(val_by_dom["instruct_prose"]) == 60 and len(val_by_dom["public_expository"]) == 60

    nr_val_sorted = sorted(val_by_dom["instruct_prose"], key=lambda pid: det_rank_key("cedar_v3_val_strat_v1", pid))
    mg_val_sorted = sorted(val_by_dom["public_expository"], key=lambda pid: det_rank_key("cedar_v3_val_strat_v1", pid))
    val_select_pids = sorted(nr_val_sorted[:30] + mg_val_sorted[:30])
    val_calib_pids = sorted(nr_val_sorted[30:] + mg_val_sorted[30:])
    assert len(set(val_select_pids) & set(val_calib_pids)) == 0

    val_manifest = {
        "protocol": "a_priori_source_stratified_30_nr_30_mage_split",
        "hash_tag": "cedar_v3_val_strat_v1",
        "val_select_parent_ids": val_select_pids,
        "val_calib_parent_ids": val_calib_pids,
        "counts": {
            "val_select_instruct_prose": 30,
            "val_select_public_expository": 30,
            "val_calib_instruct_prose": 30,
            "val_calib_public_expository": 30,
        },
    }
    atomic_save_json(os.path.join(V2_DIR, "val_manifest.json"), val_manifest)

    # 2. Collect ALL forbidden hashes from every evaluation & external split so train_v2 and test_confirm have ZERO overlap
    forbidden_hashes = set()
    forbidden_pids = set()
    for sp_k in ["val", "test_id", "test_ood_ar", "test_ood_llada"]:
        for p in parents_v1[sp_k]:
            forbidden_pids.add(p["parent_id"])
            forbidden_hashes.add(norm_prefix_hash(p["human_text"]))
            forbidden_hashes.add(exact_sha256(p["human_text"]))
            forbidden_hashes.add(norm_prefix_hash(p["prompt"]))
    for ext_f in ["external_esl_full.json", "external_mixset.json", "external_beemo.json", "external_mage_wild.json"]:
        for r in json.load(open(os.path.join(V1_DIR, ext_f))):
            forbidden_hashes.add(norm_prefix_hash(r["text"]))
            forbidden_hashes.add(exact_sha256(r["text"]))

    # Start train_v2 pools from existing v1 train parents, explicitly filtering out any upstream overlap (e.g. Beemo <- No Robots)!
    used_hashes = set(forbidden_hashes)
    nr_train_v2 = []
    mg_train_v2 = []
    dropped_v1_overlap = []
    for p in parents_v1["train"]:
        th = norm_prefix_hash(p["human_text"])
        eh = exact_sha256(p["human_text"])
        ph = norm_prefix_hash(p["prompt"])
        assert p["parent_id"] not in forbidden_pids
        if th in forbidden_hashes or eh in forbidden_hashes or ph in forbidden_hashes:
            dropped_v1_overlap.append(p["parent_id"])
            continue
        used_hashes.update([th, eh, ph])
        if p["domain"] == "instruct_prose":
            nr_train_v2.append(p)
        else:
            mg_train_v2.append(p)
    print(f"  [DEDUP] Retained {len(nr_train_v2)} NR + {len(mg_train_v2)} MAGE from v1 train (dropped {len(dropped_v1_overlap)} upstream Beemo/external overlaps: {dropped_v1_overlap})")

    rng = random.Random(SEED + 303)

    # Curate exact needed NR for train_v2 (to reach 400) and +40 NR for test_confirm
    need_nr_tr = 400 - len(nr_train_v2)
    ds_nr = load_dataset("HuggingFaceH4/no_robots", split="train", revision=prov["hf_dataset_commit_shas"]["HuggingFaceH4/no_robots"])
    allowed_cats = {"Generation", "Open QA", "Brainstorm", "Essay"}
    nr_extra = []
    for row in ds_nr:
        cat = row.get("category", "")
        if cat not in allowed_cats:
            continue
        msgs = row.get("messages", [])
        if len(msgs) < 2:
            continue
        user_msg = clean_text(msgs[0].get("content", ""))
        asst_msg = clean_text(msgs[1].get("content", ""))
        wc = word_count(asst_msg)
        if not (110 <= wc <= 280) or word_count(user_msg) < 6 or word_count(user_msg) > 120:
            continue
        if "```" in asst_msg or "def " in asst_msg or "<html" in asst_msg.lower():
            continue
        th, eh, ph = norm_prefix_hash(asst_msg), exact_sha256(asst_msg), norm_prefix_hash(user_msg)
        if th in used_hashes or eh in used_hashes or ph in used_hashes:
            continue
        used_hashes.update([th, eh, ph])
        nr_extra.append({
            "domain": "instruct_prose",
            "subdomain": cat,
            "prompt": user_msg,
            "human_text": asst_msg,
            "human_wc": wc,
        })
    rng.shuffle(nr_extra)
    assert len(nr_extra) >= need_nr_tr + 40, f"Need >={need_nr_tr + 40} extra no_robots parents, got {len(nr_extra)}"
    for i, item in enumerate(nr_extra[:need_nr_tr]):
        item["parent_id"] = f"nrv2_{i:04d}"
        nr_train_v2.append(item)
    nr_confirm = []
    for i, item in enumerate(nr_extra[need_nr_tr:need_nr_tr + 40]):
        item["parent_id"] = f"nrcf_{i:04d}"
        nr_confirm.append(item)

    # Curate +150 MAGE for train_v2 (balanced across cmv, xsum, yelp, eli5) and +40 MAGE for test_confirm
    ds_mage_tr = load_dataset("yaful/MAGE", split="train", revision=prov["hf_dataset_commit_shas"]["yaful/MAGE"])
    target_subs = ["cmv", "xsum", "yelp", "eli5"]
    mg_extra_by_sub = defaultdict(list)
    for row in ds_mage_tr:
        src = row.get("src", "")
        if row.get("label") != 1 or "human" not in src:
            continue
        sub = src.split("_")[0]
        if sub not in target_subs:
            continue
        txt = clean_text(row.get("text", ""))
        wc = word_count(txt)
        if not (110 <= wc <= 280) or "```" in txt or "http://" in txt or "https://" in txt:
            continue
        th, eh = norm_prefix_hash(txt), exact_sha256(txt)
        if th in used_hashes or eh in used_hashes:
            continue
        used_hashes.update([th, eh])
        first_sent = re.split(r"(?<=[.!?])\s+", txt)[0]
        topic_snippet = " ".join(first_sent.split()[:22]).rstrip(".,;:")
        genre_desc = {
            "cmv": "a reasoned opinion piece discussing the following premise",
            "xsum": "a factual news report covering the following topic",
            "yelp": "a detailed customer review reflecting the following experience",
            "eli5": "a clear, accessible explanation addressing the following topic",
        }[sub]
        prompt = f"Write {genre_desc} (approximately {(wc // 25) * 25} words): \"{topic_snippet}\""
        mg_extra_by_sub[sub].append({
            "domain": "public_expository",
            "subdomain": sub,
            "prompt": prompt,
            "human_text": txt,
            "human_wc": wc,
        })

    mg_extra_train = []
    mg_confirm = []
    need_mg_tr = 400 - len(mg_train_v2)
    base_q = need_mg_tr // 4
    rem_q = need_mg_tr % 4
    quotas_tr = {sub: base_q + (1 if idx < rem_q else 0) for idx, sub in enumerate(target_subs)}
    for sub in target_subs:
        rng.shuffle(mg_extra_by_sub[sub])
        q = quotas_tr[sub]
        assert len(mg_extra_by_sub[sub]) >= q + 10, f"Insufficient extra MAGE in {sub}: {len(mg_extra_by_sub[sub])}"
        mg_extra_train.extend(mg_extra_by_sub[sub][:q])
        mg_confirm.extend(mg_extra_by_sub[sub][q:q + 10])
    rng.shuffle(mg_extra_train)
    for i, item in enumerate(mg_extra_train):
        item["parent_id"] = f"mgv2_{i:04d}"
        mg_train_v2.append(item)
    for i, item in enumerate(mg_confirm):
        item["parent_id"] = f"mgcf_{i:04d}"

    assert len(nr_train_v2) == 400 and len(mg_train_v2) == 400

    # Curate +80 strictly unseen-genre human passages for Pristine Confirmation Set (0 overlap with all splits & external_mage_wild!)
    unseen_genre_targets = {
        "squad": ("unseen_wiki_expository", "an encyclopedic explanation covering"),
        "wp": ("unseen_narrative_writing", "a vivid narrative passage exploring"),
        "cnn": ("unseen_news_reportage", "a factual news report examining"),
        "pubmed": ("unseen_biomedical_qa", "a clear biomedical explanation addressing"),
        "hswag": ("unseen_procedural_prose", "a coherent procedural explanation continuing"),
        "roct": ("unseen_short_narrative", "a cohesive narrative passage expanding on"),
        "tldr": ("unseen_tech_summary", "an informative analytical passage discussing"),
    }
    unseen_buckets = defaultdict(list)
    for row in ds_mage_tr:
        src = row.get("src", "")
        if row.get("label") != 1 or "human" not in src:
            continue
        sub = src.split("_")[0]
        if sub not in unseen_genre_targets:
            continue
        txt = clean_text(row.get("text", ""))
        wc = word_count(txt)
        if not (110 <= wc <= 280) or "```" in txt or "http://" in txt or "https://" in txt:
            continue
        th, eh = norm_prefix_hash(txt), exact_sha256(txt)
        if th in used_hashes or eh in used_hashes:
            continue
        used_hashes.update([th, eh])
        dom_label, desc_prefix = unseen_genre_targets[sub]
        snip = " ".join(re.split(r"(?<=[.!?])\s+", txt)[0].split()[:20]).rstrip(".,;:")
        unseen_buckets[sub].append({
            "domain": dom_label,
            "subdomain": sub,
            "prompt": f"Write {desc_prefix} (approximately {(wc // 25) * 25} words): \"{snip}\"",
            "human_text": txt,
            "human_wc": wc,
        })

    # Balanced selection across available unseen genres to form 80 unseen-genre confirmation parents
    avail_subs = [s for s in sorted(unseen_buckets.keys()) if len(unseen_buckets[s]) > 0]
    for s in avail_subs:
        rng.shuffle(unseen_buckets[s])
        print(f"  [UNSEEN GENRE POOL] {s}: {len(unseen_buckets[s])} disjoint human passages")
    raid_confirm = []
    ptr = 0
    while len(raid_confirm) < 80 and avail_subs:
        progressed = False
        for s in avail_subs:
            if ptr < len(unseen_buckets[s]) and len(raid_confirm) < 80:
                raid_confirm.append(unseen_buckets[s][ptr])
                progressed = True
        if not progressed:
            break
        ptr += 1
    assert len(raid_confirm) == 80, f"Expected 80 unseen-genre confirm parents, got {len(raid_confirm)}"
    for i, item in enumerate(raid_confirm):
        item["parent_id"] = f"ugcf_{i:04d}"

    test_confirm_parents = nr_confirm + mg_confirm + raid_confirm
    rng.shuffle(test_confirm_parents)
    assert len(test_confirm_parents) == 160

    # 3. Deterministic a priori partition of 800 train_v2 parents into:
    #    train_fit (600 parents: 300 NR + 300 MAGE) and train_op (200 parents: 100 NR + 100 MAGE)
    nr_sorted = sorted(nr_train_v2, key=lambda p: det_rank_key("cedar_v3_train_op_split_v1", p["parent_id"]))
    mg_sorted = sorted(mg_train_v2, key=lambda p: det_rank_key("cedar_v3_train_op_split_v1", p["parent_id"]))
    train_fit_parents = nr_sorted[:300] + mg_sorted[:300]
    train_op_parents = nr_sorted[300:] + mg_sorted[300:]
    assert len(train_fit_parents) == 600 and len(train_op_parents) == 200

    # 4. Precompute shuffled, source-blocked Full-Factorial Latin Schedule on train_v2 (800 parents)
    draft_gens = ["Qwen/Qwen2.5-7B-Instruct", "HuggingFaceH4/zephyr-7b-beta"]
    edit_mods = ["Qwen/Qwen2.5-7B-Instruct", "tiiuae/Falcon3-7B-Instruct"]

    schedule = {}
    for part_name, sub_groups in [
        ("train_fit", [nr_sorted[:300], mg_sorted[:300]]),
        ("train_op", [nr_sorted[300:], mg_sorted[300:]]),
    ]:
        for grp in sub_groups:
            grp_shuf = grp.copy()
            rng.shuffle(grp_shuf)
            # Assign 8 draft cells (2 gens x 4 draft templates) cyclically -> near-exact balance!
            by_dcell = defaultdict(list)
            for idx, p in enumerate(grp_shuf):
                d_cell = idx % 8
                by_dcell[d_cell].append(p)

            for d_cell, plist in by_dcell.items():
                d_gen = draft_gens[d_cell // 4]
                d_tmpl = d_cell % 4
                perm8 = list(range(8))
                rng.shuffle(perm8)
                for k, p in enumerate(plist):
                    pid = p["parent_id"]
                    c_e1 = perm8[(k + d_cell) % 8]
                    c_e2 = perm8[(3 * k + d_cell + 1) % 8]
                    c_e3 = perm8[(5 * k + d_cell + 2) % 8]
                    schedule[pid] = {
                        "partition": part_name,
                        "domain": p["domain"],
                        "draft_generator": d_gen,
                        "draft_tmpl_idx": d_tmpl,
                        "edits": {
                            "E1": {"editor": edit_mods[c_e1 // 4], "tmpl_idx": c_e1 % 4, "register": EDIT_TEMPLATES_SEEN_V2["E1"][c_e1 % 4][0]},
                            "E2": {"editor": edit_mods[c_e2 // 4], "tmpl_idx": c_e2 % 4, "register": EDIT_TEMPLATES_SEEN_V2["E2"][c_e2 % 4][0]},
                            "E3": {"editor": edit_mods[c_e3 // 4], "tmpl_idx": c_e3 % 4, "register": EDIT_TEMPLATES_SEEN_V2["E3"][c_e3 % 4][0]},
                        },
                    }

    # Verify full-factorial 64-cell joint occupancy across all 800 train_v2 parents!
    for lv in ["E1", "E2", "E3"]:
        joint_cnt = Counter(
            (v["draft_generator"], v["draft_tmpl_idx"], v["edits"][lv]["editor"], v["edits"][lv]["tmpl_idx"])
            for v in schedule.values()
        )
        assert len(joint_cnt) == 64, f"Expected all 64 joint cells occupied at {lv}, got {len(joint_cnt)}"
        assert min(joint_cnt.values()) >= 10 and max(joint_cnt.values()) <= 15, (
            f"Imbalanced joint schedule at {lv}: min={min(joint_cnt.values())}, max={max(joint_cnt.values())}"
        )

    # Confirm schedule for 160 pristine test_confirm parents (Granite-3.1-8B drafts + Phi-4-mini bidirectional edits)
    confirm_schedule = {}
    for idx, p in enumerate(test_confirm_parents):
        pid = p["parent_id"]
        confirm_schedule[pid] = {
            "draft_generator": "ibm-granite/granite-3.1-8b-instruct",
            "draft_tmpl_idx": idx % 2,
            "edits": {
                "E1": {"editor": "microsoft/Phi-4-mini-instruct", "tmpl_idx": idx % 2, "register": EDIT_TEMPLATES_CONFIRM["E1"][idx % 2][0]},
                "E2": {"editor": "microsoft/Phi-4-mini-instruct", "tmpl_idx": (idx + 1) % 2, "register": EDIT_TEMPLATES_CONFIRM["E2"][(idx + 1) % 2][0]},
                "E3": {"editor": "microsoft/Phi-4-mini-instruct", "tmpl_idx": (idx // 2) % 2, "register": EDIT_TEMPLATES_CONFIRM["E3"][(idx // 2) % 2][0]},
            },
        }

    bundle = {
        "train_fit": train_fit_parents,
        "train_op": train_op_parents,
        "test_confirm": test_confirm_parents,
        "train_schedule": schedule,
        "confirm_schedule": confirm_schedule,
        "curate_seconds": round(time.time() - t0, 2),
    }
    atomic_save_json(os.path.join(V2_DIR, "parents_and_schedule_v2.json"), bundle)
    print(
        f"[CURATED & VERIFIED] train_fit={len(train_fit_parents)}, train_op={len(train_op_parents)}, "
        f"test_confirm={len(test_confirm_parents)} | 64/64 joint factorial cells occupied | {bundle['curate_seconds']}s"
    )
    return bundle


def get_stop_token_ids(tok, model) -> list:
    stop_ids = set()
    for src in [tok.eos_token_id, getattr(getattr(model, "generation_config", None), "eos_token_id", None)]:
        if src is not None:
            if isinstance(src, (list, tuple)):
                stop_ids.update(int(x) for x in src)
            else:
                stop_ids.add(int(src))
    for cand in ["<|im_end|>", "<|end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<|end_of_text|>"]:
        tid = tok.convert_tokens_to_ids(cand)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            stop_ids.add(tid)
    return sorted(stop_ids)


def compute_jobs_spec_hash(model_id: str, model_sha: str, jobs: list, max_new_tokens: int, temperature: float, top_p: float) -> str:
    h = hashlib.sha256()
    h.update(f"{model_id}|{model_sha}|{max_new_tokens}|{temperature}|{top_p}|".encode("utf-8"))
    for jb in sorted(jobs, key=lambda x: x["job_id"]):
        h.update(f"{jb['job_id']}|{exact_sha256(jb['user_prompt'])}|".encode("utf-8"))
    return h.hexdigest()


def run_verified_causal_lm_stage(
    cache_filename: str,
    model_id: str,
    model_sha: str,
    sub_batches: list,
    batch_size: int = 144,
):
    """
    Runs one or more job groups [(group_name, jobs, max_new_tokens, temperature, top_p), ...] in a SINGLE model load,
    verifying full cryptographic spec_hash on cache read, and retrying any unusually short generation (<35 words)
    with deterministic stepped seed/temperature logging.
    """
    cache_path = os.path.join(V2_DIR, cache_filename)
    spec_parts = [
        f"{gname}:{compute_jobs_spec_hash(model_id, model_sha, jbs, mnt, temp, tp)}"
        for (gname, jbs, mnt, temp, tp) in sub_batches
    ]
    master_spec_hash = exact_sha256("|".join(spec_parts))

    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        if cached.get("master_spec_hash") == master_spec_hash:
            print(f"  [VERIFIED SPEC CACHE HIT] {cache_filename} ({len(cached['outputs'])} outputs)", flush=True)
            return cached

    t0 = time.time()
    print(f"\n>>> Loading {model_id} @ {model_sha[:10]} for {cache_filename} (bs={batch_size}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_sha)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_sha,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    stop_ids = get_stop_token_ids(tok, model)

    outputs = {}
    retry_log = []
    group_timings = {}

    for gname, jobs, max_new_tokens, temperature, top_p in sub_batches:
        gt0 = time.time()
        indexed = sorted(enumerate(jobs), key=lambda x: len(x[1]["user_prompt"]))
        for start in range(0, len(indexed), batch_size):
            chunk = indexed[start:start + batch_size]
            chat_prompts = []
            for _, jb in chunk:
                msgs = [{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": jb["user_prompt"]}]
                try:
                    cp = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    cp = tok.apply_chat_template(
                        [{"role": "user", "content": SYS_PLAIN + "\n\n" + jb["user_prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                chat_prompts.append(cp)

            enc = tok(chat_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            in_len = enc["input_ids"].shape[1]
            torch.manual_seed(SEED + start + len(gname))
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=stop_ids,
                    use_cache=True,
                )
            decoded = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
            for (_, jb), raw_out in zip(chunk, decoded):
                clean_out = sanitize_output(raw_out)
                outputs[jb["job_id"]] = clean_out

            done = min(start + batch_size, len(indexed))
            if (start // batch_size) % 3 == 0 or done == len(indexed):
                print(f"  [{model_id}:{gname}] {done}/{len(indexed)} ({time.time() - gt0:.1f}s)", flush=True)

        # Deterministic symmetric quality retry if any output had <35 words
        short_jobs = [jb for jb in jobs if word_count(outputs[jb["job_id"]]) < 35]
        for r_round, s_jb in enumerate(short_jobs):
            msgs = [{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": s_jb["user_prompt"]}]
            try:
                cp = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                cp = tok.apply_chat_template([{"role": "user", "content": SYS_PLAIN + "\n\n" + s_jb["user_prompt"]}], tokenize=False, add_generation_prompt=True)
            enc1 = tok([cp], return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            il1 = enc1["input_ids"].shape[1]
            torch.manual_seed(SEED + 99991 + r_round)
            with torch.inference_mode():
                g1 = model.generate(
                    **enc1,
                    max_new_tokens=max_new_tokens + 40,
                    do_sample=True,
                    temperature=min(0.85, temperature + 0.1),
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=stop_ids,
                    use_cache=True,
                )
            cand = sanitize_output(tok.batch_decode(g1[:, il1:], skip_special_tokens=True)[0])
            old_wc = word_count(outputs[s_jb["job_id"]])
            if word_count(cand) > old_wc:
                outputs[s_jb["job_id"]] = cand
            retry_log.append({"job_id": s_jb["job_id"], "group": gname, "old_wc": old_wc, "new_wc": word_count(outputs[s_jb["job_id"]])})

        group_timings[gname] = round(time.time() - gt0, 2)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "master_spec_hash": master_spec_hash,
        "model_id": model_id,
        "model_commit_sha": model_sha,
        "wall_seconds": round(time.time() - t0, 2),
        "group_timings": group_timings,
        "retry_count": len(retry_log),
        "retry_log": retry_log,
        "outputs": outputs,
    }
    atomic_save_json(cache_path, payload)
    print(f"  [SAVED] {cache_filename}: {len(outputs)} outputs, retries={len(retry_log)}, time={payload['wall_seconds']}s", flush=True)
    return payload


def build_draft_jobs(bundle: dict):
    all_tr_parents = bundle["train_fit"] + bundle["train_op"]
    sched = bundle["train_schedule"]
    qwen_drafts, zephyr_drafts = [], []
    for p in all_tr_parents:
        pid = p["parent_id"]
        sc = sched[pid]
        twc = max(120, min(250, (p["human_wc"] // 20) * 20))
        tmpl = DRAFT_TEMPLATES_SEEN_V2[sc["draft_tmpl_idx"]]
        jb = {"job_id": pid, "user_prompt": tmpl.format(prompt=p["prompt"], target_wc=twc)}
        if sc["draft_generator"] == "Qwen/Qwen2.5-7B-Instruct":
            qwen_drafts.append(jb)
        else:
            zephyr_drafts.append(jb)

    granite_drafts = []
    c_sched = bundle["confirm_schedule"]
    for p in bundle["test_confirm"]:
        pid = p["parent_id"]
        twc = max(120, min(250, (p["human_wc"] // 20) * 20))
        tmpl = DRAFT_TEMPLATES_CONFIRM[c_sched[pid]["draft_tmpl_idx"]]
        granite_drafts.append({"job_id": pid, "user_prompt": tmpl.format(prompt=p["prompt"], target_wc=twc)})

    return qwen_drafts, zephyr_drafts, granite_drafts


def build_edit_jobs(bundle: dict, seen_drafts: dict, confirm_drafts: dict):
    all_tr_parents = bundle["train_fit"] + bundle["train_op"]
    sched = bundle["train_schedule"]
    qwen_edits, falcon_edits = [], []
    for p in all_tr_parents:
        pid = p["parent_id"]
        h_txt = p["human_text"]
        m_txt = seen_drafts[pid]
        for lv in ["E1", "E2", "E3"]:
            ed_info = sched[pid]["edits"][lv]
            reg_id, tmpl = EDIT_TEMPLATES_SEEN_V2[lv][ed_info["tmpl_idx"]]
            assert reg_id == ed_info["register"]
            for orig_label, src_txt in [(0, h_txt), (1, m_txt)]:
                jb = {"job_id": f"{pid}|O{orig_label}|{lv}", "user_prompt": tmpl.format(text=src_txt)}
                if ed_info["editor"] == "Qwen/Qwen2.5-7B-Instruct":
                    qwen_edits.append(jb)
                else:
                    falcon_edits.append(jb)

    phi_confirm_edits = []
    c_sched = bundle["confirm_schedule"]
    for p in bundle["test_confirm"]:
        pid = p["parent_id"]
        h_txt = p["human_text"]
        m_txt = confirm_drafts[pid]
        for lv in ["E1", "E2", "E3"]:
            ed_info = c_sched[pid]["edits"][lv]
            _, tmpl = EDIT_TEMPLATES_CONFIRM[lv][ed_info["tmpl_idx"]]
            for orig_label, src_txt in [(0, h_txt), (1, m_txt)]:
                phi_confirm_edits.append({"job_id": f"{pid}|O{orig_label}|{lv}", "user_prompt": tmpl.format(text=src_txt)})

    return qwen_edits, falcon_edits, phi_confirm_edits


def assemble_v2_splits():
    bundle = json.load(open(os.path.join(V2_DIR, "parents_and_schedule_v2.json")))
    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))

    d_qwen = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_qwen.json")))["outputs"]
    d_zeph = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_zephyr.json")))["outputs"]
    d_gran = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_granite_confirm.json")))["outputs"]
    e_qwen = json.load(open(os.path.join(V2_DIR, "v2_cache_edits_qwen.json")))["outputs"]
    e_falc = json.load(open(os.path.join(V2_DIR, "v2_cache_edits_falcon3.json")))["outputs"]
    e_phi = json.load(open(os.path.join(V2_DIR, "v2_cache_edits_phi4_confirm.json")))["outputs"]

    seen_drafts = {**d_qwen, **d_zeph}
    seen_edits = {**e_qwen, **e_falc}
    NOM_STR = {"E0": 0.0, "E1": 0.2, "E2": 0.5, "E3": 0.8}

    sched = bundle["train_schedule"]
    for part_key, plist in [("train_fit", bundle["train_fit"]), ("train_op", bundle["train_op"])]:
        rows = []
        for p in plist:
            pid = p["parent_id"]
            sc = sched[pid]
            base_map = {0: p["human_text"], 1: seen_drafts[pid]}
            for orig_label in (0, 1):
                raw_t = base_map[orig_label]
                for lv in ["E0", "E1", "E2", "E3"]:
                    if lv == "E0":
                        txt = raw_t
                        ed_mod = "none"
                        ed_tmpl = -1
                        ed_reg = 0
                    else:
                        txt = seen_edits[f"{pid}|O{orig_label}|{lv}"]
                        ed_mod = sc["edits"][lv]["editor"]
                        ed_tmpl = sc["edits"][lv]["tmpl_idx"]
                        ed_reg = sc["edits"][lv]["register"]
                    assert len(txt.strip()) > 0, f"Empty text at {pid} O{orig_label} {lv}"
                    rows.append({
                        "id": f"{pid}_O{orig_label}_{lv}",
                        "parent_id": pid,
                        "split": part_key,
                        "domain": p["domain"],
                        "subdomain": p["subdomain"],
                        "origin_label": orig_label,
                        "edit_level": lv,
                        "edit_register": ed_reg,  # 0=unedited, 1=formal_edit, 2=humanized_edit
                        "nominal_strength": NOM_STR[lv],
                        "measured_lex_dist": round(token_jaccard_dist(raw_t, txt), 4),
                        "draft_generator": "human" if orig_label == 0 else sc["draft_generator"],
                        "draft_tmpl_idx": -1 if orig_label == 0 else sc["draft_tmpl_idx"],
                        "editor_model": ed_mod,
                        "edit_tmpl_idx": ed_tmpl,
                        "text": txt,
                        "wc": word_count(txt),
                    })
        atomic_save_json(os.path.join(V2_DIR, f"split_{part_key}_v2.json"), rows)
        print(f"Saved split_{part_key}_v2.json: {len(rows)} rows")

    # Combined 800-parent split_train_v2.json for baselines / full-800 ablations
    fit_rows = json.load(open(os.path.join(V2_DIR, "split_train_fit_v2.json")))
    op_rows = json.load(open(os.path.join(V2_DIR, "split_train_op_v2.json")))
    atomic_save_json(os.path.join(V2_DIR, "split_train_v2.json"), fit_rows + op_rows)
    print(f"Saved split_train_v2.json: {len(fit_rows) + len(op_rows)} rows (800 complete octets)")

    # Assemble Pristine Confirmation Benchmark (160 unseen parents x 8 cells = 1,280 rows)
    c_sched = bundle["confirm_schedule"]
    conf_rows = []
    for p in bundle["test_confirm"]:
        pid = p["parent_id"]
        sc = c_sched[pid]
        base_map = {0: p["human_text"], 1: d_gran[pid]}
        for orig_label in (0, 1):
            raw_t = base_map[orig_label]
            for lv in ["E0", "E1", "E2", "E3"]:
                if lv == "E0":
                    txt = raw_t
                    ed_mod = "none"
                    ed_tmpl = -1
                    ed_reg = 0
                else:
                    txt = e_phi[f"{pid}|O{orig_label}|{lv}"]
                    ed_mod = sc["edits"][lv]["editor"]
                    ed_tmpl = sc["edits"][lv]["tmpl_idx"]
                    ed_reg = sc["edits"][lv]["register"]
                assert len(txt.strip()) > 0, f"Empty confirm text at {pid} O{orig_label} {lv}"
                conf_rows.append({
                    "id": f"{pid}_O{orig_label}_{lv}",
                    "parent_id": pid,
                    "split": "test_confirm",
                    "domain": p["domain"],
                    "subdomain": p["subdomain"],
                    "origin_label": orig_label,
                    "edit_level": lv,
                    "edit_register": ed_reg,
                    "nominal_strength": NOM_STR[lv],
                    "measured_lex_dist": round(token_jaccard_dist(raw_t, txt), 4),
                    "draft_generator": "human" if orig_label == 0 else sc["draft_generator"],
                    "draft_tmpl_idx": -1 if orig_label == 0 else sc["draft_tmpl_idx"],
                    "editor_model": ed_mod,
                    "edit_tmpl_idx": ed_tmpl,
                    "text": txt,
                    "wc": word_count(txt),
                })
    atomic_save_json(os.path.join(V2_DIR, "split_test_confirm.json"), conf_rows)
    print(f"Saved split_test_confirm.json: {len(conf_rows)} rows (160 complete unseen octets)")

    # Record stage timing summary
    timings = {}
    for fn in [
        "v2_cache_drafts_qwen.json",
        "v2_cache_drafts_zephyr.json",
        "v2_cache_drafts_granite_confirm.json",
        "v2_cache_edits_qwen.json",
        "v2_cache_edits_falcon3.json",
        "v2_cache_edits_phi4_confirm.json",
    ]:
        obj = json.load(open(os.path.join(V2_DIR, fn)))
        timings[fn] = {
            "wall_seconds": obj["wall_seconds"],
            "retry_count": obj["retry_count"],
            "master_spec_hash": obj["master_spec_hash"],
            "model_commit_sha": obj["model_commit_sha"],
        }
    prov["stage2_generation_timings"] = timings
    atomic_save_json(os.path.join(V2_DIR, "provenance_lock.json"), prov)
    print("=== V2 BENCHMARK & PRISTINE CONFIRMATION SPLIT ASSEMBLED & VERIFIED! ===")


def main():
    stage = os.environ.get("CEDAR_V3_STAGE", "curate")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    bs_7b = 160 if vram_gb >= 75 else 80
    bs_phi = 224 if vram_gb >= 75 else 112
    print(f"=== v3_build_v2_benchmark | stage={stage} | VRAM={vram_gb:.1f}GB ===", flush=True)

    if stage == "curate":
        build_curated_manifests()
        return

    bundle = json.load(open(os.path.join(V2_DIR, "parents_and_schedule_v2.json")))
    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]
    qwen_drafts, zephyr_drafts, granite_drafts = build_draft_jobs(bundle)

    if stage == "drafts_a":
        run_verified_causal_lm_stage(
            "v2_cache_drafts_qwen.json",
            "Qwen/Qwen2.5-7B-Instruct",
            m_shas["Qwen/Qwen2.5-7B-Instruct"],
            [("qwen_drafts", qwen_drafts, 290, 0.75, 0.92)],
            batch_size=bs_7b,
        )
        run_verified_causal_lm_stage(
            "v2_cache_drafts_granite_confirm.json",
            "ibm-granite/granite-3.1-8b-instruct",
            m_shas["ibm-granite/granite-3.1-8b-instruct"],
            [("granite_confirm_drafts", granite_drafts, 290, 0.75, 0.92)],
            batch_size=bs_7b,
        )
        return

    if stage == "drafts_b":
        run_verified_causal_lm_stage(
            "v2_cache_drafts_zephyr.json",
            "HuggingFaceH4/zephyr-7b-beta",
            m_shas["HuggingFaceH4/zephyr-7b-beta"],
            [("zephyr_drafts", zephyr_drafts, 290, 0.75, 0.92)],
            batch_size=bs_7b,
        )
        return

    d_qwen = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_qwen.json")))["outputs"]
    d_zeph = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_zephyr.json")))["outputs"]
    d_gran = json.load(open(os.path.join(V2_DIR, "v2_cache_drafts_granite_confirm.json")))["outputs"]
    qwen_edits, falcon_edits, phi_confirm_edits = build_edit_jobs(bundle, {**d_qwen, **d_zeph}, d_gran)

    if stage == "edits_a":
        run_verified_causal_lm_stage(
            "v2_cache_edits_qwen.json",
            "Qwen/Qwen2.5-7B-Instruct",
            m_shas["Qwen/Qwen2.5-7B-Instruct"],
            [("qwen_edits", qwen_edits, 290, 0.55, 0.92)],
            batch_size=bs_7b,
        )
        run_verified_causal_lm_stage(
            "v2_cache_edits_phi4_confirm.json",
            "microsoft/Phi-4-mini-instruct",
            m_shas["microsoft/Phi-4-mini-instruct"],
            [("phi4_confirm_edits", phi_confirm_edits, 290, 0.50, 0.92)],
            batch_size=bs_phi,
        )
        return

    if stage == "edits_b":
        run_verified_causal_lm_stage(
            "v2_cache_edits_falcon3.json",
            "tiiuae/Falcon3-7B-Instruct",
            m_shas["tiiuae/Falcon3-7B-Instruct"],
            [("falcon3_edits", falcon_edits, 290, 0.55, 0.92)],
            batch_size=bs_7b,
        )
        return

    if stage == "assemble":
        assemble_v2_splits()
        return

    raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
