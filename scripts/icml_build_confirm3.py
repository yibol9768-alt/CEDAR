#!/usr/bin/env python3
"""Confirm-3 (docs/CONFIRM3_SPEC_2026-11-15.md): BBC News 2023-05/06 (pre) vs 2025-05/06 (post), symmetric octets.

Stages (env CEDAR_C3_STAGE):
  curate   - local CPU: pick 150 parents per period, write data_c3/parents_and_schedule_c3.json
  part_a   - Colab GPU (40 GB ok): GLM-4-9B-0414 drafts; Apertus-8B edits E1-E3 of all human siblings and GLM drafts
  part_b   - Colab GPU (>= 48 GB): ERNIE-4.5-21B-A3B drafts; Apertus-8B edits of ERNIE drafts
  assemble - CPU: degenerate-octet filter, symmetry check, write split_test_confirm3.json
Smoke: CEDAR_C3_SMOKE=4 limits each generation group to 4 jobs and writes *_smoke caches.
Data dir: CEDAR_C3_DIR (default data_c3 next to scripts/, or /content/cedar_data_c3 on Colab)."""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # executed as a Colab cell: the bundle unpacks the helper modules into /content
    HERE = "/content"
ROOT = os.path.dirname(HERE)
ON_COLAB = os.path.isdir("/content")
C3_DIR = os.environ.get("CEDAR_C3_DIR", "/content/cedar_data_c3" if ON_COLAB else os.path.join(ROOT, "data_c3"))
os.makedirs(C3_DIR, exist_ok=True)
SEED = 20261116
SMOKE = int(os.environ.get("CEDAR_C3_SMOKE", "0"))
SUF = "_smoke" if SMOKE else ""
BUNDLE = os.path.join(C3_DIR, "parents_and_schedule_c3.json")

M_GEN = ["zai-org/GLM-4-9B-0414", "baidu/ERNIE-4.5-21B-A3B-PT"]
M_EDIT = "swiss-ai/Apertus-8B-Instruct-2509"
assert not any("deepseek" in m.lower() for m in M_GEN + [M_EDIT])
PERIODS = {"pre": ["2023-05", "2023-06"], "post": ["2025-05", "2025-06"]}
PER_GROUP = 30
GROUPS = ["sport", "biz_sci", "world", "culture", "uk"]
SEC_BIZ = {"Business", "Technology", "Science & Environment", "Health", "Science", "Tech", "Money"}
SEC_WORLD = {"World", "Europe", "US & Canada", "Asia", "Middle East", "Africa", "Latin America", "Australia", "China", "India"}
SEC_CULTURE = {"Culture", "Entertainment & Arts", "Newsbeat", "BBC InDepth", "Arts", "Entertainment"}

DRAFT_T = ["Write the following piece as plain prose of about {target_wc} words. Return only the article body, without a headline or notes:\n\n{prompt}",
           "Here is an assignment for a reporter. Deliver roughly {target_wc} words of finished copy and nothing else:\n\n{prompt}"]
EDIT_T = {  # one formal (1) and one colloquial (2) per level; wordings disjoint from v1/v2/Confirm/E2
    "E1": [(1, "Proofread the passage below: repair grammar, spelling and awkward wording so it reads as careful formal English, but leave every sentence in its place and every fact as it is. Give back only the proofread passage:\n\n{text}"),
           (2, "Lightly loosen up the passage below so it sounds like everyday speech, with contractions and plainer words, but leave every sentence in its place and every fact as it is. Give back only the passage:\n\n{text}")],
    "E2": [(1, "Restate each sentence of the passage below in sophisticated, exact language, using different words and constructions while keeping the facts and the order of sentences. Give back only the restated passage:\n\n{text}"),
           (2, "Restate each sentence of the passage below the way someone would tell it over coffee, using different words while keeping the facts and the order of sentences. Give back only the restated passage:\n\n{text}")],
    "E3": [(1, "Restructure the passage below into a carefully organised formal account: reorder and regroup its points, combine or break up sentences, and keep its main message and facts. Give back only the restructured passage:\n\n{text}"),
           (2, "Restructure the passage below into a relaxed, chatty retelling: reorder and regroup its points, mix short and long sentences, and keep its main message and facts. Give back only the restructured passage:\n\n{text}")],
}
NOM_STR = {"E0": 0.0, "E1": 0.2, "E2": 0.5, "E3": 0.8}


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- helpers identical to scripts/v3_build_v2_benchmark.py (copied so curation runs without torch) ----
def atomic_save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def clean_text(t):
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def word_count(t):
    return len(t.split())


def norm_prefix_hash(t):
    return hashlib.sha256(re.sub(r"\W+", "", t.lower())[:300].encode("utf-8")).hexdigest()[:16]


def exact_sha256(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def det_rank_key(tag, item_id):
    return hashlib.sha256(f"{tag}|{item_id}".encode("utf-8")).hexdigest()


def token_jaccard_dist(a, b):
    sa, sb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))


# ------------------------------------------------------------------------------------------------ curation
def forbidden_hashes():
    hs = set()
    for p in [x for lst in json.load(open(os.path.join(ROOT, "data/raw_parents.json"))).values() for x in lst]:
        hs.update([norm_prefix_hash(p["human_text"]), exact_sha256(p["human_text"]), norm_prefix_hash(p["prompt"])])
    b2 = json.load(open(os.path.join(ROOT, "data_v2/parents_and_schedule_v2.json")))
    for k in ("train_fit", "train_op", "test_confirm"):
        for p in b2[k]:
            hs.update([norm_prefix_hash(p["human_text"]), exact_sha256(p["human_text"]), norm_prefix_hash(p["prompt"])])
    be = json.load(open(os.path.join(ROOT, "data_e2/parents_and_schedule_e2.json")))
    for k in ("val_shift", "test_confirm2"):
        for p in be[k]:
            hs.update([norm_prefix_hash(p["human_text"]), exact_sha256(p["human_text"]), norm_prefix_hash(p["prompt"])])
    for f in ("external_esl_full.json", "external_mixset.json", "external_beemo.json", "external_mage_wild.json"):
        for r in json.load(open(os.path.join(ROOT, "data", f))):
            hs.update([norm_prefix_hash(r["text"]), exact_sha256(r["text"])])
    return hs


SKIP_PARA = ("To play this video", "This video can not be played", "Watch:", "Listen:", "Follow live", "Get in touch",
             "Sign up for", "Download the BBC", "Follow BBC", "Subscribe", "Image source", "Image caption")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘'])|(?<=[.!?][\"”’'])\s+(?=[A-Z0-9\"“‘'])")


def group_of(row):
    sec, link = (row.get("section") or "").strip(), row.get("link") or ""
    if "/sport/" in link:
        return "sport"
    if sec in SEC_BIZ:
        return "biz_sci"
    if sec in SEC_WORLD:
        return "world"
    if sec in SEC_CULTURE:
        return "culture"
    return "uk"


def extract_passage(content, target):
    """Leading paragraphs then sentences until >= target words (<= 280); None if unusable."""
    paras = []
    for p in re.split(r"\n\s*\n", content or ""):
        p = " ".join(p.split())
        if not p or any(s in p for s in SKIP_PARA):
            continue
        if not re.search(r"[.!?][\"”’']?$", p):  # captions, repeated headlines, bylines
            continue
        p = re.sub(r",? external(?=[\s.,;:!?)\"”’]|$)", "", p)
        paras.append(SENT_RE.split(p))
    out, wc, ns = [], 0, 0
    for sents in paras:
        cur = []
        for s in sents:
            n = word_count(s)
            if wc + n > 280:
                break
            cur.append(s); wc += n; ns += 1
            if wc >= target:
                break
        if cur:
            out.append(" ".join(cur))
        if wc >= target or (cur and len(cur) < len(sents)):
            break
    txt = "\n\n".join(out)
    if not (120 <= wc <= 280) or ns < 4:
        return None
    return txt


def flat(t):
    """Confirm-3 texts are whitespace-normalised to one paragraph: the 2023 and 2025 crawls mark paragraphs
    differently (7.6 vs 1.5 breaks per passage), so layout must not differ between periods or classes."""
    return " ".join((t or "").split())


def clean_title(t):
    t = re.sub(r"\s+-\s+BBC (News|Sport)(\s+\w+)?\s*$", "", (t or "").strip())
    return " ".join(t.split())


def curate():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    t0 = time.time()
    used = forbidden_hashes()
    ds_id = "RealTimeData/bbc_news_alltime"
    ds_sha = HfApi().dataset_info(ds_id).sha
    rng = random.Random(SEED)
    stats, parents = {}, []
    for per, months in PERIODS.items():
        pools, cnt = defaultdict(list), Counter()
        for mon in months:
            for row in load_dataset(ds_id, mon, split="train", revision=ds_sha):
                cnt["rows"] += 1
                link, title = row.get("link") or "", clean_title(row.get("title"))
                desc = " ".join((row.get("description") or "").split())
                if "/live/" in link or (row.get("section") or "") == "The Papers" or not title or len(desc) < 20:
                    cnt["skip_meta"] += 1; continue
                if not str(row.get("published_date") or "").startswith(mon):
                    cnt["skip_date"] += 1; continue
                lo_hi = det_rank_key(f"c3len|{per}", link)
                target = 130 + int(lo_hi[:8], 16) % 131  # deterministic per-article target in [130, 260]
                txt = extract_passage(row.get("content"), target)
                if txt is None:
                    cnt["skip_extract"] += 1; continue
                if "http" in txt or "BBC" in txt or "bbc" in txt.lower().split():
                    cnt["skip_outlet"] += 1; continue
                hs = [norm_prefix_hash(txt), exact_sha256(txt), norm_prefix_hash(title)]
                if any(h in used for h in hs):
                    cnt["skip_dup"] += 1; continue
                used.update(hs)
                wc = word_count(txt)
                pools[group_of(row)].append({
                    "domain": "bbc_news", "subdomain": group_of(row), "period": per, "month": mon,
                    "published_date": str(row.get("published_date")), "section": row.get("section"), "link": link,
                    "human_text": flat(txt), "human_wc": wc,
                    "prompt": f"Write a news report of about {(wc // 25) * 25} words with the headline \"{title}\" and the summary line \"{desc}\"."})
        for g in GROUPS:
            assert len(pools[g]) >= PER_GROUP, f"{per}/{g}: only {len(pools[g])}"
            pools[g].sort(key=lambda p: det_rank_key(f"c3pick|{per}|{g}", p["link"]))
            parents += pools[g][:PER_GROUP]
        stats[per] = {"filter_counts": dict(cnt), "pool_sizes": {g: len(pools[g]) for g in GROUPS}}
    rng.shuffle(parents)
    for per in PERIODS:
        k = 0
        for p in parents:
            if p["period"] == per:
                p["parent_id"] = f"c3{per}_{k:03d}"; k += 1
    sch = {}
    for i, p in enumerate(sorted(parents, key=lambda q: det_rank_key("c3", q["parent_id"]))):
        sch[p["parent_id"]] = {"draft_generator": M_GEN[i % 2], "draft_tmpl_idx": (i // 2) % 2,
                               "edits": {lv: {"register": EDIT_T[lv][(i + j) % 2][0], "tmpl_idx": (i + j) % 2}
                                         for j, lv in enumerate(["E1", "E2", "E3"])}}
    atomic_save_json(BUNDLE, {"dataset": ds_id, "dataset_sha": ds_sha, "test_confirm3": parents, "c3_schedule": sch,
                              "stats": stats, "curate_seconds": round(time.time() - t0, 1)})
    gen_by = Counter((p["period"], sch[p["parent_id"]]["draft_generator"]) for p in parents)
    say(f"[curate] {len(parents)} parents; {json.dumps(stats)}; generator split {dict(gen_by)}")


# ------------------------------------------------------------------------------------------------ generation
def ensure_transformers():
    """Apertus needs transformers>=4.56; check the installed version before anything imports transformers."""
    from importlib.metadata import version
    v = tuple(int(x) for x in re.findall(r"\d+", version("transformers"))[:2])
    if v < (4, 57):
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57"])
        say(f"upgraded transformers from {v}")


def jobs_drafts():
    b = json.load(open(BUNDLE))
    by_gen = defaultdict(list)
    for p in b["test_confirm3"]:
        sc = b["c3_schedule"][p["parent_id"]]
        twc = max(120, min(250, (int(p["human_wc"]) // 20) * 20))
        by_gen[sc["draft_generator"]].append({"job_id": p["parent_id"], "target_wc": twc,
                                              "user_prompt": DRAFT_T[sc["draft_tmpl_idx"]].format(prompt=p["prompt"], target_wc=twc)})
    return {g: (sorted(v, key=lambda j: j["job_id"])[:SMOKE] if SMOKE else sorted(v, key=lambda j: j["job_id"])) for g, v in by_gen.items()}


def strip_title(t):
    """Drafts sometimes open with a headline line despite the instruction (seen in the smoke test, 11-15 09:2x SGT,
    before any detector score): drop a first line that has no terminal punctuation and at most 25 words."""
    lines = (t or "").strip().split("\n")
    first = lines[0].strip().strip("#*").strip()
    if len(lines) > 1 and first and not re.search(r"[.!?:][\"\u201d\u2019')]?$", first) and len(first.split()) <= 25:
        return "\n".join(lines[1:]).strip()
    return t


def load_drafts():
    d = {}
    for i in range(len(M_GEN)):
        fp = os.path.join(C3_DIR, f"c3_drafts_g{i}{SUF}.json")
        if os.path.exists(fp):
            d.update({k: flat(strip_title(v)) for k, v in json.load(open(fp))["outputs"].items()})
    return d


def jobs_edits(drafts, human=True, gens=None):
    """Edit jobs for human siblings (all parents in scope) and for machine siblings whose draft exists and whose
    generator is in `gens` (None = all). In smoke mode the scope is the parents of the smoke draft jobs."""
    b = json.load(open(BUNDLE))
    scope = {j["job_id"] for v in jobs_drafts().values() for j in v}
    jobs = []
    for p in b["test_confirm3"]:
        pid = p["parent_id"]
        if pid not in scope:
            continue
        gen = b["c3_schedule"][pid]["draft_generator"]
        srcs = ([(0, p["human_text"])] if human else []) + \
               ([(1, drafts[pid])] if pid in drafts and (gens is None or gen in gens) else [])
        for o, src in srcs:
            for lv in ("E1", "E2", "E3"):
                ti = b["c3_schedule"][pid]["edits"][lv]["tmpl_idx"]
                jobs.append({"job_id": f"{pid}|O{o}|{lv}", "user_prompt": EDIT_T[lv][ti][1].format(text=src)})
    return jobs


def run_gen(stage):
    """part_a (fits a 40 GB card): GLM drafts + edits of all human siblings and of GLM drafts.
    part_b (needs ~48 GB): ERNIE drafts + edits of ERNIE drafts."""
    ensure_transformers()
    sys.path.insert(0, HERE)
    import v3_build_e2_splits as e2  # GPU helpers (gen_causal); redirect its cache dir and suffix
    e2.E2_DIR, e2.SUF = C3_DIR, SUF
    j = jobs_drafts()
    gi = 0 if stage == "part_a" else 1
    e2.gen_causal(f"c3_drafts_g{gi}", M_GEN[gi], j[M_GEN[gi]], 450, 0.75, bs=64)
    e2.gen_causal(f"c3_edits_{stage[-1]}", M_EDIT, jobs_edits(load_drafts(), human=(stage == "part_a"), gens={M_GEN[gi]}), 450, 0.5, bs=96)


# ------------------------------------------------------------------------------------------------ assembly
REFUSAL_RE = re.compile(r"(i'?m sorry|i am sorry|i cannot|i can'?t|as an ai|sorry, but|please provide)", re.I)


def degenerate(t, min_words=35):  # identical to the E2 rule
    w = t.lower().split()
    if len(w) < min_words or REFUSAL_RE.match(t.strip()):
        return True
    tri = list(zip(w, w[1:], w[2:]))
    return len(set(tri)) < 0.5 * len(tri)


def assemble():
    b = json.load(open(BUNDLE))
    drafts = load_drafts()
    edits = {}
    for part in ("a", "b"):
        edits.update(json.load(open(os.path.join(C3_DIR, f"c3_edits_{part}{SUF}.json")))["outputs"])
    rows, dropped, dropped_outlet = [], [], []
    for p in b["test_confirm3"]:
        pid = p["parent_id"]
        if pid not in drafts:
            continue
        sc = b["c3_schedule"][pid]
        cells = {(o, lv): (raw if lv == "E0" else flat(edits.get(f"{pid}|O{o}|{lv}", "")))
                 for o, raw in ((0, p["human_text"]), (1, drafts[pid])) for lv in ("E0", "E1", "E2", "E3")}
        if any(degenerate(t) for t in cells.values()):
            dropped.append(pid); continue
        if any(re.search(r"\bBBC\b", t) for t in cells.values()):  # outlet name only on the machine side = label cue
            dropped_outlet.append(pid); continue
        for (o, lv), txt in cells.items():
            raw = p["human_text"] if o == 0 else drafts[pid]
            rows.append({"id": f"{pid}_O{o}_{lv}", "parent_id": pid, "split": "test_confirm3", "domain": p["domain"],
                         "subdomain": p["subdomain"], "period": p["period"], "origin_label": o, "edit_level": lv,
                         "edit_register": 0 if lv == "E0" else sc["edits"][lv]["register"], "nominal_strength": NOM_STR[lv],
                         "measured_lex_dist": round(token_jaccard_dist(raw, txt), 4),
                         "draft_generator": "human" if o == 0 else sc["draft_generator"],
                         "draft_tmpl_idx": -1 if o == 0 else sc["draft_tmpl_idx"],
                         "editor_model": "none" if lv == "E0" else M_EDIT,
                         "edit_tmpl_idx": -1 if lv == "E0" else sc["edits"][lv]["tmpl_idx"], "text": txt, "wc": word_count(txt)})
    key = defaultdict(dict)
    for r in rows:
        key[(r["parent_id"], r["edit_level"])][r["origin_label"]] = (r["editor_model"], r["edit_tmpl_idx"], r["edit_register"])
    assert all(len(v) == 2 and v[0] == v[1] for v in key.values())
    atomic_save_json(os.path.join(C3_DIR, f"split_test_confirm3{SUF}.json"), rows)
    per = Counter(r["period"] for r in rows if r["edit_level"] == "E0" and r["origin_label"] == 0)
    say(f"[assemble] {len(rows)} rows, {len(rows) // 8} octets {dict(per)}, dropped {len(dropped)} degenerate: {dropped}; "
        f"{len(dropped_outlet)} naming the outlet: {dropped_outlet}")


if __name__ == "__main__":
    st = os.environ.get("CEDAR_C3_STAGE", "curate")
    say(f"=== Confirm-3 stage={st} smoke={SMOKE} dir={C3_DIR} ===")
    if st == "curate":
        curate()
    elif st in ("part_a", "part_b"):
        run_gen(st)
    elif st == "assemble":
        assemble()
    else:
        raise ValueError(st)
