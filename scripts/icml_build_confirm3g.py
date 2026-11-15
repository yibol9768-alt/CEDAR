#!/usr/bin/env python3
"""Confirm-3G (docs/CONFIRM3G_SPEC_2026-11-15.md): Gemini 3.8 Flash edits the Confirm-3 drafts at E1-E3.
Stages: python scripts/icml_build_confirm3g.py run [N_PARENTS]   -> data_c3g/c3g_edits.json (resumable cache)
        python scripts/icml_build_confirm3g.py assemble           -> data_c3g/split_test_confirm3g.json
Calls `agy -p` (Antigravity CLI, headless) in an empty directory, at most 3 at a time."""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from icml_build_confirm3 import NOM_STR, REFUSAL_RE, atomic_save_json, degenerate, flat, token_jaccard_dist, word_count  # noqa: E402

C3 = os.path.join(ROOT, "data_c3", "split_test_confirm3.json")
OUT = os.path.join(ROOT, "data_c3g"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "c3g_edits.json")
MODEL = "gemini-3.8-flash-medium"
EDITOR = f"{MODEL} (Antigravity CLI)"
WORKDIR = os.environ.get("C3G_WORKDIR", os.path.join(OUT, "agy_empty")); os.makedirs(WORKDIR, exist_ok=True)
AGY = os.path.expanduser("~/.local/bin/agy")
EDIT_T = {  # register 1 = formal, 2 = colloquial; user-style wordings disjoint from all earlier splits
    "E1": {1: "Can you fix the grammar, spelling and punctuation in this? Keep my wording and keep every sentence where it is. Send back only the corrected text.",
           2: "Quick proofread please: fix any typos and grammar slips but keep it casual, and don't move or merge my sentences. Send back only the fixed text."},
    "E2": {1: "Please reword this sentence by sentence in clearer, more polished language. Same facts, same order. Send back only the reworded text.",
           2: "Rewrite this in your own words so it sounds natural and easygoing, sentence by sentence, same facts and same order. Send back only the rewritten text."},
    "E3": {1: "Improve this piece: reorganise it so it flows like a well-edited article, merging or splitting sentences as needed, and keep all the facts. Send back only the improved version.",
           2: "Make this more engaging and relaxed. Feel free to reorganise it and change the sentences however you like, but keep the facts. Send back only the new version."},
}
LOCK = threading.Lock()


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def octets():
    rows = json.load(open(C3)); by = defaultdict(dict)
    for r in rows:
        by[r["parent_id"]][(r["origin_label"], r["edit_level"])] = r
    return rows, by


def jobs(n_parents=None):
    _, by = octets(); out = []
    for pid in sorted(by)[: n_parents or None]:
        for o in (0, 1):
            src = by[pid][(o, "E0")]["text"]
            for lv in ("E1", "E2", "E3"):
                reg = by[pid][(0, lv)]["edit_register"]; assert reg == by[pid][(1, lv)]["edit_register"]
                out.append({"job_id": f"{pid}|O{o}|{lv}", "register": reg, "prompt": f"{EDIT_T[lv][reg]}\n\n{src}"})
    return out


def call(prompt):
    for attempt in range(3):
        try:
            p = subprocess.run([AGY, "-p", prompt, "--model", MODEL, "--output-format", "json", "--disable-slash-commands",
                                "--print-timeout", "240s"], cwd=WORKDIR, capture_output=True, text=True, timeout=300)
            d = json.loads(p.stdout.strip().splitlines()[-1])
            if d.get("status") == "SUCCESS" and d.get("response", "").strip():
                return {"text": d["response"], "conversation_id": d.get("conversation_id"), "attempts": attempt + 1,
                        "output_tokens": d.get("usage", {}).get("output_tokens")}
        except Exception as e:  # noqa: BLE001
            say(f"call failed ({attempt + 1}/3): {type(e).__name__}: {str(e)[:120]}")
        time.sleep(5 * (attempt + 1))
    return None


def run(n_parents=None):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {"editor": EDITOR, "outputs": {}, "failed": []}
    todo = [j for j in jobs(n_parents) if j["job_id"] not in cache["outputs"]]
    say(f"{len(todo)} calls to make ({len(cache['outputs'])} cached)"); t0 = time.time(); done = 0
    with ThreadPoolExecutor(3) as ex:
        fut = {ex.submit(call, j["prompt"]): j for j in todo}
        for f in as_completed(fut):
            j, r = fut[f], f.result(); done += 1
            with LOCK:
                if r is None:
                    cache["failed"].append(j["job_id"])
                else:
                    cache["outputs"][j["job_id"]] = r
                if done % 15 == 0 or done == len(todo):
                    atomic_save_json(CACHE, cache)
                    say(f"{done}/{len(todo)} done, {len(cache['failed'])} failed, {(time.time() - t0) / done:.1f}s/call")
    atomic_save_json(CACHE, cache)


PREAMBLE = re.compile(r"^\s*(here|sure|certainly)\b[^\n]*:\s*\n", re.I)


def clean(t):
    t = PREAMBLE.sub("", t, count=1)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M)
    return flat(t)


def assemble():
    rows, by = octets()
    cache = json.load(open(CACHE))["outputs"]
    out, dropped = [], Counter()
    for pid in sorted(by):
        cells = {}
        for o in (0, 1):
            cells[(o, "E0")] = by[pid][(o, "E0")]["text"]
            for lv in ("E1", "E2", "E3"):
                r = cache.get(f"{pid}|O{o}|{lv}")
                cells[(o, lv)] = clean(r["text"]) if r else None
        if any(t is None for t in cells.values()):
            dropped["missing"] += 1; continue
        if any(degenerate(t) for t in cells.values()):
            dropped["degenerate"] += 1; continue
        if any(re.search(r"\bBBC\b", t) for t in cells.values()):
            dropped["outlet"] += 1; continue
        for (o, lv), txt in cells.items():
            base = by[pid][(o, "E0")]
            reg = 0 if lv == "E0" else by[pid][(0, lv)]["edit_register"]
            out.append({**{k: base[k] for k in ("parent_id", "domain", "subdomain", "period", "origin_label", "draft_generator", "draft_tmpl_idx")},
                        "id": f"{pid}_O{o}_{lv}", "split": "test_confirm3g", "edit_level": lv, "edit_register": reg,
                        "nominal_strength": NOM_STR[lv], "measured_lex_dist": round(token_jaccard_dist(base["text"], txt), 4),
                        "editor_model": "none" if lv == "E0" else EDITOR, "edit_tmpl_idx": -1 if lv == "E0" else reg - 1,
                        "text": txt, "wc": word_count(txt)})
    key = defaultdict(dict)
    for r in out:
        key[(r["parent_id"], r["edit_level"])][r["origin_label"]] = (r["editor_model"], r["edit_tmpl_idx"], r["edit_register"])
    assert all(len(v) == 2 and v[0] == v[1] for v in key.values())
    atomic_save_json(os.path.join(OUT, "split_test_confirm3g.json"), out)
    ld = defaultdict(list)
    for r in out:
        if r["edit_level"] != "E0":
            ld[(r["origin_label"], r["edit_level"])].append(r["measured_lex_dist"])
    say(f"{len(out)} rows, {len(out) // 8} octets; dropped {dict(dropped)}; mean lex dist (human/machine) " +
        ", ".join(f"{lv} {sum(ld[(0, lv)]) / len(ld[(0, lv)]):.2f}/{sum(ld[(1, lv)]) / len(ld[(1, lv)]):.2f}" for lv in ("E1", "E2", "E3")))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif st == "assemble":
        assemble()
    else:
        raise ValueError(st)
