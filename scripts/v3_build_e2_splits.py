#!/usr/bin/env python3
"""E2 data: `val_shift` (selection) and `test_confirm2` (final, untouched) octet splits.

Plan and rules: docs/E2_plan_2026-11-15.md.  Runs on Colab (paths under /content).
Stages (env CEDAR_E2_STAGE): curate | val_drafts | val_edits | assemble_val | c2_drafts | c2_edits | assemble_c2
Smoke test: CEDAR_E2_SMOKE=4 limits every generation group to 4 jobs and writes *_smoke caches.
"""
import gc
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_build_v2_benchmark import (  # noqa: E402  (helpers only; module guards its main())
    SYS_PLAIN, atomic_save_json, clean_text, det_rank_key, exact_sha256, get_stop_token_ids, norm_prefix_hash,
    sanitize_output, token_jaccard_dist, word_count,
)

V1_DIR = "/content/cedar_data"
V2_DIR = "/content/cedar_data_v2"
E2_DIR = "/content/cedar_data_e2"
os.makedirs(E2_DIR, exist_ok=True)
SEED = 20261115
SMOKE = int(os.environ.get("CEDAR_E2_SMOKE", "0"))
SUF = "_smoke" if SMOKE else ""

M_VAL_AR = "mistralai/Mistral-7B-Instruct-v0.3"
M_VAL_DIFF = "GSAI-ML/LLaDA-8B-Instruct"
M_VAL_EDIT = "HuggingFaceTB/SmolLM3-3B"
M_C2_AR = "openai/gpt-oss-20b"
M_C2_DIFF = "Dream-org/Dream-v0-Instruct-7B"
M_C2_EDIT = "01-ai/Yi-1.5-9B-Chat"
ALL_MODELS = [M_VAL_AR, M_VAL_DIFF, M_VAL_EDIT, M_C2_AR, M_C2_DIFF, M_C2_EDIT]
assert not any("deepseek" in m.lower() for m in ALL_MODELS)

DRAFT_T = {
    "val": ["Please write a passage of about {target_wc} words responding to the request below. Reply with the passage only:\n\n{prompt}",
            "Compose roughly {target_wc} words of prose for the following task, returning nothing but the text:\n\n{prompt}"],
    "c2": ["Produce a passage of around {target_wc} words for the request that follows. Give only the passage, no title or notes:\n\n{prompt}",
           "Write about {target_wc} words in response to the following, as finished prose with nothing else:\n\n{prompt}"],
}
EDIT_T = {  # level -> [(register, template)], one formal and one colloquial per level, disjoint from v1/v2/confirm
    "val": {
        "E1": [(1, "Correct grammar, spelling and punctuation in the text below and smooth any clumsy phrasing in a polished register. Do not reorder sentences or change content. Return the corrected text only:\n\n{text}"),
               (2, "Make the text below sound a little more relaxed and spoken, with contractions and plain words, without reordering sentences or changing any fact. Return the text only:\n\n{text}")],
        "E2": [(1, "Rephrase every sentence of the text below in refined, precise prose with new vocabulary and syntax, keeping all facts and their order. Return the rephrased text only:\n\n{text}"),
               (2, "Rephrase every sentence of the text below the way a person would say it casually to a friend, keeping all facts and their order. Return the rephrased text only:\n\n{text}")],
        "E3": [(1, "Rework the structure of the text below into tightly organised, formal prose: change the order of points, merge or split sentences, keep the main claims and facts. Return the rewritten text only:\n\n{text}"),
               (2, "Rework the text below into a loose, chatty explanation: change the order of points, vary sentence length, use everyday transitions, keep the main claims and facts. Return the rewritten text only:\n\n{text}")],
    },
    "c2": {
        "E1": [(1, "Edit the following text for correctness and formal polish only: fix grammar and wording while keeping each sentence where it is and every detail unchanged. Output the edited text and nothing else:\n\n{text}"),
               (2, "Edit the following text so it reads as natural, informal writing: use contractions and simpler words while keeping each sentence where it is and every detail unchanged. Output the edited text and nothing else:\n\n{text}")],
        "E2": [(1, "Express each sentence of the following text again in elevated, carefully worded prose, changing words and sentence structure but not the facts or their order. Output the new text and nothing else:\n\n{text}"),
               (2, "Express each sentence of the following text again in a down-to-earth, conversational voice with short and long sentences mixed, changing words but not the facts or their order. Output the new text and nothing else:\n\n{text}")],
        "E3": [(1, "Reorganise the following text as a well-structured formal piece: regroup and reorder its points and restructure sentences, preserving its central argument and facts. Output the new text and nothing else:\n\n{text}"),
               (2, "Reorganise the following text as a lively, informal first-person style explanation: regroup and reorder its points and loosen the sentences, preserving its central argument and facts. Output the new text and nothing else:\n\n{text}")],
    },
}
NOM_STR = {"E0": 0.0, "E1": 0.2, "E2": 0.5, "E3": 0.8}
MAGE_DESC = {"cmv": "a reasoned opinion piece discussing the following premise",
             "xsum": "a factual news report covering the following topic",
             "yelp": "a detailed customer review reflecting the following experience",
             "eli5": "a clear, accessible explanation addressing the following topic"}
RAID_DESC = {"books": "Write a plot summary (approximately {wc} words) of the book titled: \"{title}\"",
             "recipes": "Write a recipe with ingredients and directions (approximately {wc} words) for: \"{title}\"",
             "wiki": "Write an encyclopedia-style introduction (approximately {wc} words) about: \"{title}\"",
             "news": "Write a news article (approximately {wc} words) with the headline: \"{title}\""}


def prov():
    return json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))


# ----------------------------------------------------------------------------------------------- curation
def forbidden_hashes():
    hs = set()
    raw = json.load(open(os.path.join(V1_DIR, "raw_parents.json")))
    for k, lst in raw.items():
        for p in lst:
            hs.update([norm_prefix_hash(p["human_text"]), exact_sha256(p["human_text"]), norm_prefix_hash(p["prompt"])])
    b2 = json.load(open(os.path.join(V2_DIR, "parents_and_schedule_v2.json")))
    for k in ("train_fit", "train_op", "test_confirm"):
        for p in b2[k]:
            hs.update([norm_prefix_hash(p["human_text"]), exact_sha256(p["human_text"]), norm_prefix_hash(p["prompt"])])
    for f in ("external_esl_full.json", "external_mixset.json", "external_beemo.json", "external_mage_wild.json"):
        for r in json.load(open(os.path.join(V1_DIR, f))):
            hs.update([norm_prefix_hash(r["text"]), exact_sha256(r["text"])])
    return hs


def curate():
    from datasets import load_dataset
    t0 = time.time()
    used = forbidden_hashes()
    pv = prov()["hf_dataset_commit_shas"]
    rng = random.Random(SEED)

    # val_shift: 30 No Robots + 30 MAGE (same filters as v1)
    nr = []
    for row in load_dataset("HuggingFaceH4/no_robots", split="train", revision=pv["HuggingFaceH4/no_robots"]):
        if row.get("category") not in {"Generation", "Open QA", "Brainstorm", "Essay"} or len(row["messages"]) < 2:
            continue
        u, a = clean_text(row["messages"][0]["content"]), clean_text(row["messages"][1]["content"])
        wc = word_count(a)
        if not (110 <= wc <= 280) or not (6 <= word_count(u) <= 120) or "```" in a or "def " in a or "<html" in a.lower():
            continue
        hs = [norm_prefix_hash(a), exact_sha256(a), norm_prefix_hash(u)]
        if any(h in used for h in hs):
            continue
        used.update(hs)
        nr.append({"domain": "instruct_prose", "subdomain": row["category"], "prompt": u, "human_text": a, "human_wc": wc})
    mg = defaultdict(list)
    for row in load_dataset("yaful/MAGE", split="train", revision=pv["yaful/MAGE"]):
        src = row.get("src", "")
        if row.get("label") != 1 or "human" not in src:
            continue
        sub = src.split("_")[0]
        if sub not in MAGE_DESC:
            continue
        txt = clean_text(row.get("text", ""))
        wc = word_count(txt)
        if not (110 <= wc <= 280) or "```" in txt or "http" in txt:
            continue
        hs = [norm_prefix_hash(txt), exact_sha256(txt)]
        if any(h in used for h in hs):
            continue
        used.update(hs)
        snip = " ".join(re.split(r"(?<=[.!?])\s+", txt)[0].split()[:22]).rstrip(".,;:")
        mg[sub].append({"domain": "public_expository", "subdomain": sub, "human_text": txt, "human_wc": wc,
                        "prompt": f"Write {MAGE_DESC[sub]} (approximately {(wc // 25) * 25} words): \"{snip}\""})
    rng.shuffle(nr)
    val_nr = nr[:30]
    val_mg = []
    for sub in sorted(mg):
        rng.shuffle(mg[sub])
    while len(val_mg) < 30:
        for sub in sorted(mg):
            if len(val_mg) < 30 and mg[sub]:
                val_mg.append(mg[sub].pop())
    val_parents = val_nr + val_mg
    for i, p in enumerate(val_parents):
        p["parent_id"] = f"vs_{i:03d}"

    # test_confirm2: 50 RAID human passages from each of 4 domains
    want = {d: 50 for d in RAID_DESC}
    pools = defaultdict(list)
    ds = load_dataset("liamdugan/raid", "raid", split="train", streaming=True, revision=pv["liamdugan/raid"])
    n_seen = 0
    for row in ds:
        n_seen += 1
        d = row.get("domain")
        if row.get("model") != "human" or row.get("attack") != "none" or d not in want:
            continue
        if len(pools[d]) >= 4 * want[d]:
            if all(len(pools[k]) >= 4 * want[k] for k in want):
                break
            continue
        txt, title = clean_text(row.get("generation", "")), clean_text(row.get("title", "") or "")
        wc = word_count(txt)
        if not (110 <= wc <= 280) or len(title) < 4 or "http" in txt:
            continue
        hs = [norm_prefix_hash(txt), exact_sha256(txt)]
        if any(h in used for h in hs):
            continue
        used.update(hs)
        pools[d].append({"domain": f"raid_{d}", "subdomain": d, "human_text": txt, "human_wc": wc,
                         "prompt": RAID_DESC[d].format(wc=(wc // 25) * 25, title=title)})
    c2 = []
    for d in sorted(want):
        assert len(pools[d]) >= want[d], f"RAID {d}: only {len(pools[d])}"
        rng.shuffle(pools[d])
        c2.extend(pools[d][:want[d]])
    rng.shuffle(c2)
    for i, p in enumerate(c2):
        p["parent_id"] = f"c2_{i:03d}"

    def schedule(parents, tag, gens):
        sch = {}
        for i, p in enumerate(sorted(parents, key=lambda q: det_rank_key(tag, q["parent_id"]))):
            sch[p["parent_id"]] = {"draft_generator": gens[i % 2], "draft_tmpl_idx": (i // 2) % 2,
                                   "edits": {lv: {"register": EDIT_T[tag][lv][(i + j) % 2][0], "tmpl_idx": (i + j) % 2}
                                             for j, lv in enumerate(["E1", "E2", "E3"])}}
        return sch
    bundle = {"val_shift": val_parents, "test_confirm2": c2,
              "val_schedule": schedule(val_parents, "val", [M_VAL_AR, M_VAL_DIFF]),
              "c2_schedule": schedule(c2, "c2", [M_C2_AR, M_C2_DIFF]),
              "raid_rows_streamed": n_seen, "curate_seconds": round(time.time() - t0, 1)}
    atomic_save_json(os.path.join(E2_DIR, "parents_and_schedule_e2.json"), bundle)
    print(f"[curate] val_shift={len(val_parents)} confirm2={len(c2)} raid_rows={n_seen} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------------------- generators
def model_sha(mid):
    from huggingface_hub import HfApi
    p = prov()["hf_model_commit_shas"]
    return p.get(mid) or HfApi().model_info(mid).sha


def target_wc(p, upper=250):
    return max(120, min(upper, (int(p["human_wc"]) // 20) * 20))


def limit(jobs):
    return jobs[:SMOKE] if SMOKE else jobs


REFUSAL_RE = re.compile(r"(i'?m sorry|i am sorry|i cannot|i can'?t|as an ai|sorry, but|please provide)", re.I)


TRAIL_RE = re.compile(r"(?<=[.!?\"\u201d])(?:\s+\(?\d+(?:\s*(?:words?|paragraphs?))?\)?\.?)+\s*$", re.I)


def strip_len_artifacts(t):
    """Dream with a minimum length sometimes pads the end with the requested length ("... 200 words. 1")."""
    return TRAIL_RE.sub("", t).rstrip()


def degenerate(t, min_words=35):
    """Too short, or looping (fewer than half of the word trigrams distinct). Fixed before any E2 result."""
    w = t.lower().split()
    if len(w) < min_words or REFUSAL_RE.match(t.strip()):
        return True
    tri = list(zip(w, w[1:], w[2:]))
    return len(set(tri)) < 0.5 * len(tri)


def gen_causal(cache, mid, jobs, max_new, temp, top_p=0.92, bs=96, chat_kwargs=None, system=SYS_PLAIN, final_channel=False):
    path = os.path.join(E2_DIR, cache + SUF + ".json")
    if os.path.exists(path):
        print(f"[cache] {path}", flush=True)
        return json.load(open(path))["outputs"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sha = model_sha(mid)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid, revision=sha)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, revision=sha, dtype="auto" if final_channel else torch.bfloat16,
                                                 device_map="cuda").eval()
    stop = get_stop_token_ids(tok, model)
    if final_channel:  # harmony: <|end|> closes the analysis message; only <|return|>/<|call|> end the turn
        stop = [tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")]
    out, short, raws = {}, [], {}
    order = sorted(jobs, key=lambda j: len(j["user_prompt"]))
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]
        texts = []
        for jb in chunk:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": jb["user_prompt"]}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {})))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to("cuda")
        torch.manual_seed(SEED + s)
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p,
                               pad_token_id=tok.pad_token_id, eos_token_id=stop)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=not final_channel)
        for jb, raw in zip(chunk, dec):
            if final_channel:  # gpt-oss harmony format: keep only the final channel
                raws[jb["job_id"]] = raw
                m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", raw, re.S)
                raw = m.group(1) if m else ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
            out[jb["job_id"]] = sanitize_output(raw)
            if word_count(out[jb["job_id"]]) < 35:
                short.append(jb["job_id"])
        print(f"  [{mid}] {min(s + bs, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    atomic_save_json(path, {"model_id": mid, "model_commit_sha": sha, "max_new_tokens": max_new, "temperature": temp,
                            "short_outputs": short, "wall_seconds": round(time.time() - t0, 1), "outputs": out,
                            **({"raw_harmony": raws} if final_channel else {})})
    print(f"[saved] {path}: {len(out)} outputs, {len(short)} short", flush=True)
    return out


def _gumbel(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_(min=1e-12)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _n_transfer(mask_index, steps):
    n = mask_index.sum(dim=1, keepdim=True)
    base, rem = n // steps, n % steps
    t = torch.zeros(n.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(n.size(0)):
        t[i, :rem[i]] += 1
    return t


def load_remote_model(mid, sha):
    """trust_remote_code models (LLaDA, Dream) written for transformers 4.x: their tie_weights() takes no
    arguments, but transformers 5 passes missing_keys/recompute_mapping. Wrap it; behaviour is unchanged."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(mid, revision=sha, trust_remote_code=True)
    for k, v in (("use_cache", False), ("output_attentions", False), ("output_hidden_states", False)):
        if k not in cfg.to_dict():
            setattr(cfg, k, v)  # 4.x defaults the remote code reads but transformers 5 no longer sets
    cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], mid, revision=sha)
    from transformers import modeling_rope_utils as mru
    if "default" not in mru.ROPE_INIT_FUNCTIONS:  # removed in transformers 5; standard RoPE inverse frequencies
        def _default_rope(config, device=None, seq_len=None, **kw):
            dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
            inv = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            return inv, 1.0
        mru.ROPE_INIT_FUNCTIONS["default"] = _default_rope
    if not getattr(cls, "_e2_tie_patched", False):
        orig = cls.tie_weights
        cls.tie_weights = lambda self, *a, **k: orig(self)
        cls._e2_tie_patched = True
    import transformers
    dkw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return cls.from_pretrained(mid, revision=sha, config=cfg, **{dkw: torch.bfloat16}).cuda().eval()


def gen_llada(cache, jobs, bs=24):
    path = os.path.join(E2_DIR, cache + SUF + ".json")
    if os.path.exists(path):
        return json.load(open(path))["outputs"]
    from transformers import AutoModel, AutoTokenizer
    sha = model_sha(M_VAL_DIFF)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(M_VAL_DIFF, revision=sha, trust_remote_code=True)
    tok.padding_side = "left"
    model = load_remote_model(M_VAL_DIFF, sha)
    out = {}
    mask_id, steps, gen_len, blk = 126336, 64, 256, 64
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        chat = [tok.apply_chat_template([{"role": "user", "content": j["user_prompt"]}], add_generation_prompt=True, tokenize=False) for j in chunk]
        enc = tok(chat, padding=True, truncation=True, max_length=256, return_tensors="pt", add_special_tokens=False).to("cuda")
        B, L = enc["input_ids"].shape
        x = torch.full((B, L + gen_len), mask_id, dtype=torch.long, device="cuda")
        x[:, :L] = enc["input_ids"]
        attn = torch.cat([enc["attention_mask"], torch.ones((B, gen_len), dtype=enc["attention_mask"].dtype, device="cuda")], -1)
        torch.manual_seed(SEED + s)
        with torch.inference_mode():
            for nb in range(gen_len // blk):
                b0, b1 = L + nb * blk, L + (nb + 1) * blk
                ntr = _n_transfer(x[:, b0:b1] == mask_id, steps // (gen_len // blk))
                for i in range(steps // (gen_len // blk)):
                    mi = x == mask_id
                    logits = model(x, attention_mask=attn).logits
                    x0 = torch.argmax(_gumbel(logits, 0.2), -1)
                    p = torch.gather(F.softmax(logits.float(), -1), -1, x0.unsqueeze(-1)).squeeze(-1)
                    p[:, b1:] = -np.inf
                    x0 = torch.where(mi, x0, x)
                    conf = torch.where(mi, p, torch.tensor(-np.inf, device="cuda", dtype=p.dtype))
                    tr = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(B):
                        k = int(ntr[j, i].item())
                        if k > 0:
                            tr[j, torch.topk(conf[j], k=k).indices] = True
                    x[tr] = x0[tr]
        for jb, o in zip(chunk, tok.batch_decode(x[:, L:], skip_special_tokens=True)):
            out[jb["job_id"]] = sanitize_output(o)
        print(f"  [LLaDA] {min(s + bs, len(jobs))}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    atomic_save_json(path, {"model_id": M_VAL_DIFF, "model_commit_sha": sha, "steps": steps, "gen_length": gen_len,
                            "wall_seconds": round(time.time() - t0, 1), "outputs": out})
    return out


def gen_dream(cache, jobs, bs=32):
    path = os.path.join(E2_DIR, cache + SUF + ".json")
    if os.path.exists(path):
        return json.load(open(path))["outputs"]
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5 and os.environ.get("CEDAR_E2_DREAM_CHILD") != "1":
        # Dream's remote code (modeling + generation_utils) targets transformers 4.46.2, its pinned version;
        # run this generator alone in a child process with that version on PYTHONPATH.
        import subprocess
        tdir = "/content/tf4462"
        if not os.path.exists(os.path.join(tdir, "transformers")):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", tdir, "transformers==4.46.2"], check=True)
        env = dict(os.environ, CEDAR_E2_DREAM_CHILD="1", CEDAR_E2_STAGE="c2_dream_child",
                   PYTHONPATH=tdir + os.pathsep + os.environ.get("PYTHONPATH", ""))
        subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env=env, check=True)
        return json.load(open(path))["outputs"]
    from transformers import AutoTokenizer
    print(f"  [Dream] transformers {transformers.__version__}", flush=True)
    sha = model_sha(M_C2_DIFF)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(M_C2_DIFF, revision=sha, trust_remote_code=True)
    tok.padding_side = "left"
    model = load_remote_model(M_C2_DIFF, sha)
    out, retries = {}, {}
    todo = list(jobs)
    DREAM_MAX = 384
    end_ids = sorted({i for i in [tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                  tok.convert_tokens_to_ids("<|endoftext|>")] if isinstance(i, int) and i >= 0})
    print(f"  [Dream] end ids {end_ids}", flush=True)
    DREAM_TEMPS = [0.2, 0.5, 0.8]  # README setting first; near-greedy loops repeat identically, so retries sample hotter
    for attempt in range(3):  # re-sample drafts that come out degenerate (looping/short), new seed each round
      for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        msgs = [[{"role": "user", "content": j["user_prompt"]}] for j in chunk]
        enc = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True)
        ids, am = enc.input_ids.cuda(), enc.attention_mask.cuda()
        torch.manual_seed(SEED + s + 100003 * attempt)
        P = ids.shape[1]
        min_tok = torch.tensor([min(int(1.1 * j["target_wc"]), DREAM_MAX - 40) for j in chunk], device="cuda")
        pos = torch.arange(DREAM_MAX, device="cuda")[None, :]
        block = (pos < min_tok[:, None])  # [B, DREAM_MAX]: generated positions where the answer may not end yet

        def min_len_hook(step, x, logits):  # masked diffusion analogue of min_new_tokens
            lg = logits[:, P:P + DREAM_MAX]
            for t in end_ids:
                lg[..., t] = torch.where(block, torch.finfo(lg.dtype).min, lg[..., t])
            return logits
        with torch.inference_mode():
            o = model.diffusion_generate(ids, attention_mask=am, max_new_tokens=DREAM_MAX, output_history=False,
                                         return_dict_in_generate=True, steps=DREAM_MAX, temperature=DREAM_TEMPS[attempt],
                                         top_p=0.95, alg="entropy", alg_temp=0.0,
                                         generation_logits_hook_func=min_len_hook)
        for jb, seq in zip(chunk, o.sequences):
            txt = tok.decode(seq[ids.shape[1]:].tolist())
            txt = txt.split(tok.eos_token)[0] if tok.eos_token else txt
            out[jb["job_id"]] = strip_len_artifacts(sanitize_output(re.sub(r"<\|[^>]*\|>", "", txt)))
            retries[jb["job_id"]] = attempt
        print(f"  [Dream] attempt {attempt} {min(s + bs, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
      todo = [j for j in todo if degenerate(out[j["job_id"]])]
      if not todo:
          break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    atomic_save_json(path, {"model_id": M_C2_DIFF, "model_commit_sha": sha, "steps": DREAM_MAX, "max_new_tokens": DREAM_MAX, "min_new_tokens_rule": "1.1*target_wc (end/pad suppressed before)", "temperatures_by_attempt": DREAM_TEMPS,
                            "wall_seconds": round(time.time() - t0, 1), "attempt_index": retries,
                            "still_degenerate": [j["job_id"] for j in todo], "outputs": out})
    return out


# ----------------------------------------------------------------------------------------------- jobs
def bundle():
    return json.load(open(os.path.join(E2_DIR, "parents_and_schedule_e2.json")))


def draft_jobs(tag):
    b = bundle()
    parents = b["val_shift"] if tag == "val" else b["test_confirm2"]
    sch = b["val_schedule"] if tag == "val" else b["c2_schedule"]
    by_gen = defaultdict(list)
    for p in parents:
        sc = sch[p["parent_id"]]
        twc = target_wc(p, 220 if "LLaDA" in sc["draft_generator"] or "Dream" in sc["draft_generator"] else 250)
        by_gen[sc["draft_generator"]].append({"job_id": p["parent_id"], "target_wc": twc,
                                              "user_prompt": DRAFT_T[tag][sc["draft_tmpl_idx"]].format(prompt=p["prompt"], target_wc=twc)})
    return {g: limit(sorted(v, key=lambda j: j["job_id"])) for g, v in by_gen.items()}


def edit_jobs(tag, drafts):
    b = bundle()
    parents = b["val_shift"] if tag == "val" else b["test_confirm2"]
    sch = b["val_schedule"] if tag == "val" else b["c2_schedule"]
    jobs = []
    for p in parents:
        pid = p["parent_id"]
        if pid not in drafts:
            continue
        for o, src in ((0, p["human_text"]), (1, drafts[pid])):
            for lv in ("E1", "E2", "E3"):
                ti = sch[pid]["edits"][lv]["tmpl_idx"]
                jobs.append({"job_id": f"{pid}|O{o}|{lv}", "user_prompt": EDIT_T[tag][lv][ti][1].format(text=src)})
    return jobs


def load_drafts(tag):
    names = ["e2_val_drafts_mistral", "e2_val_drafts_llada"] if tag == "val" else ["e2_c2_drafts_gptoss", "e2_c2_drafts_dream"]
    d = {}
    for n in names:
        d.update(json.load(open(os.path.join(E2_DIR, n + SUF + ".json")))["outputs"])
    return d


def assemble(tag):
    b = bundle()
    parents = b["val_shift"] if tag == "val" else b["test_confirm2"]
    sch = b["val_schedule"] if tag == "val" else b["c2_schedule"]
    drafts = load_drafts(tag)
    edits = json.load(open(os.path.join(E2_DIR, f"e2_{tag}_edits{SUF}.json")))["outputs"]
    editor = M_VAL_EDIT if tag == "val" else M_C2_EDIT
    split = "val_shift" if tag == "val" else "test_confirm2"
    rows, dropped = [], []
    for p in parents:
        pid = p["parent_id"]
        if pid not in drafts:
            continue
        sc = sch[pid]
        cells = {}
        for o, raw in ((0, p["human_text"]), (1, drafts[pid])):
            for lv in ("E0", "E1", "E2", "E3"):
                cells[(o, lv)] = raw if lv == "E0" else edits.get(f"{pid}|O{o}|{lv}", "")
        if any(degenerate(t) for t in cells.values()):  # drop whole octet so symmetry is preserved
            dropped.append(pid)
            continue
        for (o, lv), txt in cells.items():
            raw = p["human_text"] if o == 0 else drafts[pid]
            rows.append({"id": f"{pid}_O{o}_{lv}", "parent_id": pid, "split": split, "domain": p["domain"],
                         "subdomain": p["subdomain"], "origin_label": o, "edit_level": lv,
                         "edit_register": 0 if lv == "E0" else sc["edits"][lv]["register"], "nominal_strength": NOM_STR[lv],
                         "measured_lex_dist": round(token_jaccard_dist(raw, txt), 4),
                         "draft_generator": "human" if o == 0 else sc["draft_generator"],
                         "draft_tmpl_idx": -1 if o == 0 else sc["draft_tmpl_idx"],
                         "editor_model": "none" if lv == "E0" else editor,
                         "edit_tmpl_idx": -1 if lv == "E0" else sc["edits"][lv]["tmpl_idx"], "text": txt, "wc": word_count(txt)})
    # exact symmetry check: editor/template/register identical across origins for every (parent, level)
    key = defaultdict(dict)
    for r in rows:
        key[(r["parent_id"], r["edit_level"])][r["origin_label"]] = (r["editor_model"], r["edit_tmpl_idx"], r["edit_register"])
    assert all(len(v) == 2 and v[0] == v[1] for v in key.values())
    atomic_save_json(os.path.join(E2_DIR, f"split_{split}{SUF}.json"), rows)
    print(f"[assemble] {split}: {len(rows)} rows, {len(rows) // 8} octets, dropped {len(dropped)}: {dropped}", flush=True)


def main():
    stage = os.environ.get("CEDAR_E2_STAGE", "curate")
    print(f"=== E2 stage={stage} smoke={SMOKE} ===", flush=True)
    if stage == "curate":
        curate()
    elif stage == "val_drafts":
        j = draft_jobs("val")
        gen_causal("e2_val_drafts_mistral", M_VAL_AR, j[M_VAL_AR], 290, 0.75)
        gen_llada("e2_val_drafts_llada", j[M_VAL_DIFF])
    elif stage == "val_edits":
        gen_causal("e2_val_edits", M_VAL_EDIT, edit_jobs("val", load_drafts("val")), 450, 0.5,
                   chat_kwargs={"enable_thinking": False})
    elif stage == "c2_drafts":
        j = draft_jobs("c2")
        gen_causal("e2_c2_drafts_gptoss", M_C2_AR, j[M_C2_AR], 900, 0.75, bs=64,
                   chat_kwargs={"reasoning_effort": "low"}, final_channel=True)
        gen_dream("e2_c2_drafts_dream", j[M_C2_DIFF])
    elif stage == "c2_dream_child":
        gen_dream("e2_c2_drafts_dream", draft_jobs("c2")[M_C2_DIFF])
    elif stage == "c2_edits":
        gen_causal("e2_c2_edits", M_C2_EDIT, edit_jobs("c2", load_drafts("c2")), 450, 0.5)
    elif stage == "assemble_val":
        assemble("val")
    elif stage == "assemble_c2":
        assemble("c2")
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    main()
