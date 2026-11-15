import os
import re
import ast
import json
import random
import hashlib
import warnings
from collections import Counter, defaultdict
from datasets import load_dataset

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

SEED = 20261114
rng = random.Random(SEED)

OUT_DIR = "/content/cedar_data"
os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse excessive blank lines while preserving normal paragraphs
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def word_count(t: str) -> int:
    return len(t.split())


def text_hash(t: str) -> str:
    norm = re.sub(r"\W+", "", t.lower())[:300]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


print("[1/5] Curating Seen Domain 1: HuggingFaceH4/no_robots ...")
ds_nr = load_dataset("HuggingFaceH4/no_robots", split="train")
seen_hashes = set()
dom1_pool = []
allowed_cats = {"Generation", "Open QA", "Brainstorm", "Essay"}

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
    # Exclude code-heavy snippets so prose discourse metrics are clean
    if "```" in asst_msg or "def " in asst_msg or "<html" in asst_msg.lower():
        continue
    h = text_hash(asst_msg)
    ph = text_hash(user_msg)
    if h in seen_hashes or ph in seen_hashes:
        continue
    seen_hashes.add(h)
    seen_hashes.add(ph)
    dom1_pool.append({
        "parent_id": f"nr_{len(dom1_pool):04d}",
        "domain": "instruct_prose",
        "subdomain": cat,
        "prompt": user_msg,
        "human_text": asst_msg,
        "human_wc": wc,
    })

rng.shuffle(dom1_pool)
assert len(dom1_pool) >= 410, f"Expected >=410 no_robots parents, got {len(dom1_pool)}"
dom1_train = dom1_pool[:250]
dom1_val = dom1_pool[250:310]
dom1_test = dom1_pool[310:410]
print(f"  Domain 1 pool={len(dom1_pool)} -> train=250, val=60, test_id=100")

print("[2/5] Curating Seen Domain 2: yaful/MAGE (human expository: cmv, xsum, yelp, eli5) ...")
ds_mage_tr = load_dataset("yaful/MAGE", split="train")
dom2_by_sub = defaultdict(list)
target_subs = {"cmv", "xsum", "yelp", "eli5"}

for row in ds_mage_tr:
    src = row.get("src", "")
    # Verified: in MAGE, label == 1 and 'human' in src denotes authentic human text
    if row.get("label") != 1 or "human" not in src:
        continue
    sub = src.split("_")[0]
    if sub not in target_subs:
        continue
    txt = clean_text(row.get("text", ""))
    wc = word_count(txt)
    if not (110 <= wc <= 280):
        continue
    if "```" in txt or "http://" in txt or "https://" in txt:
        continue
    h = text_hash(txt)
    if h in seen_hashes:
        continue
    seen_hashes.add(h)
    # Extract neutral topic cue from opening clause (first 18 words) without leaking full human draft
    first_sent = re.split(r"(?<=[.!?])\s+", txt)[0]
    topic_snippet = " ".join(first_sent.split()[:22]).rstrip(".,;:")
    genre_desc = {
        "cmv": "a reasoned opinion piece discussing the following premise",
        "xsum": "a factual news report covering the following topic",
        "yelp": "a detailed customer review reflecting the following experience",
        "eli5": "a clear, accessible explanation addressing the following topic",
    }[sub]
    prompt = f"Write {genre_desc} (approximately { (wc // 25) * 25 } words): \"{topic_snippet}\""
    dom2_by_sub[sub].append({
        "domain": "public_expository",
        "subdomain": sub,
        "prompt": prompt,
        "human_text": txt,
        "human_wc": wc,
    })

dom2_selected = []
for sub in sorted(target_subs):
    rng.shuffle(dom2_by_sub[sub])
    print(f"  MAGE human {sub}: {len(dom2_by_sub[sub])} available")
    dom2_selected.extend(dom2_by_sub[sub][:110])

rng.shuffle(dom2_selected)
for idx, item in enumerate(dom2_selected):
    item["parent_id"] = f"mg_{idx:04d}"

assert len(dom2_selected) >= 410, f"Expected >=410 MAGE parents, got {len(dom2_selected)}"
dom2_train = dom2_selected[:250]
dom2_val = dom2_selected[250:310]
dom2_test = dom2_selected[310:410]
print(f"  Domain 2 selected={len(dom2_selected)} -> train=250, val=60, test_id=100")

print("[3/5] Curating Strictly Held-Out Unseen Domain 3: Scientific Abstracts (RAID human) ...")
ds_raid = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
dom3_pool = []
for row in ds_raid:
    if row.get("model") == "human" and row.get("domain") == "abstracts" and row.get("attack") == "none":
        txt = clean_text(row.get("generation", ""))
        title = clean_text(row.get("title", ""))
        wc = word_count(txt)
        if 110 <= wc <= 280 and len(title) > 10:
            h = text_hash(txt)
            if h not in seen_hashes:
                seen_hashes.add(h)
                dom3_pool.append({
                    "parent_id": f"abs_{len(dom3_pool):04d}",
                    "domain": "scientific_abstracts",
                    "subdomain": "raid_abstracts",
                    "prompt": f"Write a formal scientific research abstract (approximately {(wc // 25) * 25} words) for a paper titled: \"{title}\"",
                    "human_text": txt,
                    "human_wc": wc,
                })
    if len(dom3_pool) >= 360:
        break

# If RAID abstracts had <360 in range, top up from MAGE 'sci_human' (scientific abstracts)
if len(dom3_pool) < 360:
    for row in ds_mage_tr:
        src = row.get("src", "")
        if row.get("label") == 1 and src == "sci_human":
            txt = clean_text(row.get("text", ""))
            wc = word_count(txt)
            if 110 <= wc <= 280:
                h = text_hash(txt)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    first_sent = re.split(r"(?<=[.!?])\s+", txt)[0]
                    topic_snippet = " ".join(first_sent.split()[:20])
                    dom3_pool.append({
                        "parent_id": f"abs_{len(dom3_pool):04d}",
                        "domain": "scientific_abstracts",
                        "subdomain": "sci_abstracts",
                        "prompt": f"Write a formal scientific research abstract (approximately {(wc // 25) * 25} words) investigating: \"{topic_snippet}\"",
                        "human_text": txt,
                        "human_wc": wc,
                    })
        if len(dom3_pool) >= 360:
            break

rng.shuffle(dom3_pool)
assert len(dom3_pool) >= 350, f"Expected >=350 scientific abstract parents, got {len(dom3_pool)}"
dom3_ood_ar = dom3_pool[:200]
dom3_ood_llada = dom3_pool[200:350]
print(f"  Unseen Domain 3 (Scientific Abstracts): ood_ar=200, ood_diffusion_llada=150")

parents_bundle = {
    "train": dom1_train + dom2_train,       # 500 parents
    "val": dom1_val + dom2_val,             # 120 parents
    "test_id": dom1_test + dom2_test,       # 200 parents
    "test_ood_ar": dom3_ood_ar,             # 200 parents (unseen domain + unseen AR + unseen editor)
    "test_ood_llada": dom3_ood_llada,       # 150 parents (unseen domain + LLaDA-8B diffusion + unseen editor)
}
for k, v in parents_bundle.items():
    rng.shuffle(v)

with open(os.path.join(OUT_DIR, "raw_parents.json"), "w") as f:
    json.dump(parents_bundle, f, indent=2)
print("  Saved raw_parents.json successfully!")

print("[4/5] Preparing External Fairness Cohort (IELTS Non-Native Student Essays) ...")
ds_ielts = load_dataset("chillies/IELTS-writing-task-2-evaluation", split="train")
esl_pool = []
esl_hashes = set()
for row in ds_ielts:
    txt = clean_text(row.get("essay", ""))
    prm = clean_text(row.get("prompt", ""))
    wc = word_count(txt)
    if 140 <= wc <= 320 and len(prm) > 15:
        h = text_hash(txt)
        if h not in esl_hashes and h not in seen_hashes:
            esl_hashes.add(h)
            band_raw = clean_text(str(row.get("band", "")))[:4]
            esl_pool.append({
                "id": f"esl_{len(esl_pool):04d}",
                "prompt": prm,
                "text": txt,
                "origin_label": 0,  # Authentic human ESL student
                "band": band_raw,
                "wc": wc,
            })
    if len(esl_pool) >= 600:
        break

rng.shuffle(esl_pool)
esl_selected = esl_pool[:300]
with open(os.path.join(OUT_DIR, "external_esl_raw.json"), "w") as f:
    json.dump(esl_selected, f, indent=2)
print(f"  Saved external_esl_raw.json ({len(esl_selected)} independent ESL student essays)")

print("[5/5] Preparing Independent External Wild Benchmarks (MixSet-test, Beemo, MAGE-wild) ...")
# A. MixSet official test split (600 revised + deduplicated originals)
ds_ms = load_dataset("ONE-Lab/MixSet", split="test")
mixset_eval = []
orig_seen = set()
for i, r in enumerate(ds_ms):
    cat = r["mixset_category"]
    mod = str(r["model"])
    orig_txt = clean_text(r["original"])
    rev_txt = clean_text(r["revised"])
    # Verified: mod == 'None' => initial draft was Human (O=0); mod != 'None' => initial draft was Machine (O=1)
    is_machine_origin = 0 if mod == "None" else 1
    is_completion = "complete" in cat
    oh = text_hash(orig_txt)
    ms_parent_id = f"ms_orig_{oh}"
    mixset_eval.append({
        "id": f"mixset_rev_{i:04d}",
        "text": rev_txt,
        "origin_label": is_machine_origin,
        "any_ai_label": 1,
        "subset": cat,
        "is_completion": is_completion,
        "source_model": mod,
        "wc": word_count(rev_txt),
        "parent_id": ms_parent_id,
    })
    if oh not in orig_seen and word_count(orig_txt) >= 50:
        orig_seen.add(oh)
        mixset_eval.append({
            "id": f"mixset_orig_{i:04d}",
            "text": orig_txt,
            "origin_label": is_machine_origin,
            "any_ai_label": is_machine_origin,
            "subset": "raw_human_orig" if is_machine_origin == 0 else f"raw_mgt_orig_{mod}",
            "is_completion": False,
            "source_model": mod,
            "wc": word_count(orig_txt),
            "parent_id": ms_parent_id,
        })

assert len(mixset_eval) == 720, f"Expected 720 MixSet records, got {len(mixset_eval)}"
assert len({x["parent_id"] for x in mixset_eval}) == 120, "Expected 120 MixSet source families"
with open(os.path.join(OUT_DIR, "external_mixset.json"), "w") as f:
    json.dump(mixset_eval, f, indent=2)
print(f"  Saved external_mixset.json ({len(mixset_eval)} items)")

# B. Toloka Beemo (human_output, model_output, human_edits, gpt-4o/llama-70b edits)
ds_bm = load_dataset("toloka/beemo", split="train")
beemo_rows = list(ds_bm)
rng.shuffle(beemo_rows)
beemo_eval = []
used_bm_prompts = set()
for r in beemo_rows:
    pid = r.get("prompt_id", "")
    if pid in used_bm_prompts:
        continue
    h_txt = clean_text(r.get("human_output", ""))
    m_txt = clean_text(r.get("model_output", ""))
    he_txt = clean_text(r.get("human_edits", ""))
    if word_count(h_txt) < 60 or word_count(m_txt) < 60 or word_count(he_txt) < 60:
        continue
    llm_ed_txt = ""
    raw_gpt = r.get("gpt-4o_edits", "")
    try:
        parsed = ast.literal_eval(raw_gpt) if isinstance(raw_gpt, str) else raw_gpt
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            llm_ed_txt = clean_text(list(parsed[0].values())[0])
    except Exception:
        pass
    if word_count(llm_ed_txt) < 60:
        continue
    used_bm_prompts.add(pid)
    idx = len(used_bm_prompts)
    beemo_eval.extend([
        {"id": f"bm_hum_{idx:04d}", "prompt_id": pid, "text": h_txt, "origin_label": 0, "subset": "human_raw", "model": "human", "wc": word_count(h_txt)},
        {"id": f"bm_mgt_{idx:04d}", "prompt_id": pid, "text": m_txt, "origin_label": 1, "subset": "mgt_raw", "model": r.get("model", ""), "wc": word_count(m_txt)},
        {"id": f"bm_exp_{idx:04d}", "prompt_id": pid, "text": he_txt, "origin_label": 1, "subset": "mgt_expert_human_edited", "model": r.get("model", ""), "wc": word_count(he_txt)},
        {"id": f"bm_gpt4o_{idx:04d}", "prompt_id": pid, "text": llm_ed_txt, "origin_label": 1, "subset": "mgt_gpt4o_edited", "model": r.get("model", ""), "wc": word_count(llm_ed_txt)},
    ])
    if len(used_bm_prompts) >= 300:
        break

with open(os.path.join(OUT_DIR, "external_beemo.json"), "w") as f:
    json.dump(beemo_eval, f, indent=2)
print(f"  Saved external_beemo.json ({len(beemo_eval)} items across {len(used_bm_prompts)} unique prompts)")

# C. MAGE Wild Test Split (unseen domains: wp, roct, tldr, hswag, squad)
ds_mage_te = load_dataset("yaful/MAGE", split="test")
mage_h, mage_m = [], []
wild_subs = {"wp", "roct", "tldr", "hswag", "squad"}
for r in ds_mage_te:
    src = r.get("src", "")
    sub = src.split("_")[0]
    if sub not in wild_subs:
        continue
    txt = clean_text(r.get("text", ""))
    wc = word_count(txt)
    if not (80 <= wc <= 320):
        continue
    if r.get("label") == 1 and "human" in src:
        mage_h.append({"id": f"mgw_h_{len(mage_h):05d}", "text": txt, "origin_label": 0, "subset": src, "domain": sub, "wc": wc})
    elif r.get("label") == 0 and "machine" in src:
        mage_m.append({"id": f"mgw_m_{len(mage_m):05d}", "text": txt, "origin_label": 1, "subset": src, "domain": sub, "wc": wc})

rng.shuffle(mage_h)
rng.shuffle(mage_m)
mage_wild_eval = mage_h[:500] + mage_m[:500]
rng.shuffle(mage_wild_eval)
with open(os.path.join(OUT_DIR, "external_mage_wild.json"), "w") as f:
    json.dump(mage_wild_eval, f, indent=2)
print(f"  Saved external_mage_wild.json ({len(mage_wild_eval)} items: 500 human + 500 wild multi-LLM)")
print("ALL STEP 1 DATA SOURCES READY!")
