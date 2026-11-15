import os
import re
import gc
import time
import json
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUT_DIR = "/content/cedar_data"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 20261114
rng = random.Random(SEED)
torch.manual_seed(SEED)


def atomic_save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# =====================================================================
# 1. Official LLaDA-8B Masked Diffusion Batched Sampler (GSAI-ML/LLaDA)
# =====================================================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_(min=1e-12)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.inference_mode()
def llada_generate_batch(model, tokenizer, prompts_text, steps=64, gen_length=256, block_length=64, temperature=0.2, mask_id=126336):
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts_text
    ]
    enc = tokenizer(chat_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
    prompt = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :L] = prompt
    attn = torch.cat([attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        b_start = L + num_block * block_length
        b_end = L + (num_block + 1) * block_length
        block_mask_index = (x[:, b_start:b_end] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attn, use_cache=False).logits
            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)
            p = F.softmax(logits.to(torch.float32), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, b_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device, dtype=x0_p.dtype))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k_val = int(num_transfer[j, i].item())
                if k_val > 0:
                    _, sel = torch.topk(confidence[j], k=k_val)
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    outs = tokenizer.batch_decode(x[:, L:], skip_special_tokens=True)
    return [sanitize_output(o) for o in outs]


# =====================================================================
# 2. Output Sanitizer (Identical Cleaning Rule Across All Classes)
# =====================================================================
PREAMBLE_RE = re.compile(
    r"^(?:sure[,!.]?|certainly[,!.]?|here\s+is\s+[^:\n]{0,60}:|revised\s+(?:passage|text|version)\s*:|"
    r"edited\s+(?:passage|text|version)\s*:|paraphrased\s+(?:passage|text|version)\s*:|"
    r"summary\s*:)\s*",
    re.IGNORECASE,
)


def sanitize_output(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t).strip()
    for _ in range(2):
        t_new = PREAMBLE_RE.sub("", t).strip()
        if t_new != t:
            t = t_new
        else:
            break
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "“" and t[-1] == "”")):
        t = t[1:-1].strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# =====================================================================
# 3. Prompt Templates (Seen Pool vs Strictly Disjoint Held-Out Pool)
# =====================================================================
SYS_PLAIN = (
    "You are an expert writer and editor. Follow the user's instruction faithfully and output ONLY "
    "the requested passage text itself, with no introductory remarks, titles, meta-commentary, or markdown fences."
)

DRAFT_TEMPLATES_SEEN = [
    "{prompt}\n\nWrite a natural, coherent passage of roughly {target_wc} words. Output only the passage.",
    "Respond directly to the following writing prompt in approximately {target_wc} words, outputting only the text itself:\n\n{prompt}",
]

DRAFT_TEMPLATES_OOD = [
    "Compose a well-structured response of around {target_wc} words for the prompt below. Provide only the body text without headers or prefatory notes:\n\n{prompt}",
]

EDIT_TEMPLATES_SEEN = {
    "E1": [
        "Lightly copyedit the following passage for grammar, spelling, punctuation, and smooth local fluency. Keep the exact sentence order, factual claims, examples, and tone intact. Output only the copyedited passage:\n\n{text}",
        "Proofread and lightly polish grammar and awkward phrasing in the text below without changing its sentence sequence, arguments, or specific details. Return only the polished text:\n\n{text}",
        "Act as a careful copyeditor: fix grammatical issues and minor wording stiffness in this passage while strictly preserving every sentence's position, meaning, and structure. Output only the edited passage:\n\n{text}",
    ],
    "E2": [
        "Paraphrase the following passage at the sentence level using fresh vocabulary and varied clause structures, while keeping all underlying points, entities, and paragraph progression intact. Output only the paraphrased passage:\n\n{text}",
        "Rewrite each sentence of the passage below with substantially different wording and syntax while preserving every factual claim and the overall sequence of ideas. Return only the rewritten text:\n\n{text}",
        "Rephrase the text below in natural prose so lexical choices and local sentence constructions differ throughout, without adding or deleting substantive points or reordering the argument. Output only the rephrased passage:\n\n{text}",
    ],
    "E3": [
        "Substantially reorganize and rewrite the passage below: reorder points where natural, merge or split sentences, and reshape the rhetorical flow while retaining the core thesis and key facts. Output only the restructured passage:\n\n{text}",
        "Restructure the discourse organization of the following text by rearranging sentence order, combining or dividing clauses, and recasting transitions while keeping the central content accurate. Return only the rewritten passage:\n\n{text}",
    ],
    "E4_sum": [
        "Summarize the core thesis and main takeaway of the following passage in strictly TWO concise factual sentences. Output only the two-sentence summary:\n\n{text}",
    ],
    "E4_regen": [
        "Expand the following two-sentence summary into a complete, self-contained prose passage of approximately {target_wc} words. Output only the passage:\n\n{summary}",
    ],
}

EDIT_TEMPLATES_OOD = {
    "E1": [
        "Perform a conservative copyedit on the passage below—correcting only grammar, mechanics, and minor awkwardness while leaving sentence order, claims, and voice unchanged. Provide only the final text:\n\n{text}",
        "Clean up grammatical errors and minor phrasing rough edges in the text below while keeping its exact sentence-by-sentence progression and content unchanged. Output just the edited text:\n\n{text}",
    ],
    "E2": [
        "Recast the wording and syntactic phrasing of every sentence in the passage below while faithfully preserving all factual statements and the order in which ideas appear. Provide only the recast passage:\n\n{text}",
        "Transform the lexical and syntactic surface of the text below into natural alternative phrasing without altering its factual propositions or discourse order. Output just the transformed text:\n\n{text}",
    ],
}


# =====================================================================
# 4. Multi-EOS Fast Batched Causal LM Runner
# =====================================================================
def get_stop_token_ids(tok, model) -> list:
    stop_ids = set()
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, (list, tuple)):
            stop_ids.update(int(x) for x in tok.eos_token_id)
        else:
            stop_ids.add(int(tok.eos_token_id))
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_eos is not None:
        if isinstance(gen_eos, (list, tuple)):
            stop_ids.update(int(x) for x in gen_eos)
        else:
            stop_ids.add(int(gen_eos))
    for cand in ["<|im_end|>", "<|end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        tid = tok.convert_tokens_to_ids(cand)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            stop_ids.add(tid)
    return sorted(stop_ids)


def load_causal_lm(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    stop_ids = get_stop_token_ids(tok, model)
    return model, tok, stop_ids


def run_loaded_jobs(model_id: str, model, tok, stop_ids: list, jobs: list, batch_size: int, max_new_tokens: int, temperature: float):
    if not jobs:
        return {}
    t0 = time.time()
    indexed_jobs = sorted(enumerate(jobs), key=lambda x: len(x[1]["user_prompt"]))
    results = {}
    for start in range(0, len(indexed_jobs), batch_size):
        chunk = indexed_jobs[start:start + batch_size]
        chat_prompts = []
        for _, jb in chunk:
            msgs = [
                {"role": "system", "content": SYS_PLAIN},
                {"role": "user", "content": jb["user_prompt"]},
            ]
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
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=0.92 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=stop_ids,
                use_cache=True,
            )
        decoded = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
        for (_, jb), out_txt in zip(chunk, decoded):
            results[jb["job_id"]] = sanitize_output(out_txt)

        done = min(start + batch_size, len(indexed_jobs))
        if (start // batch_size) % 3 == 0 or done == len(indexed_jobs):
            print(f"  [{model_id}] {done}/{len(indexed_jobs)} ({time.time() - t0:.1f}s)", flush=True)
    return results


def run_causal_lm_jobs(model_id: str, jobs: list, batch_size: int = 96, max_new_tokens: int = 300, temperature: float = 0.7):
    if not jobs:
        return {}
    print(f"\n>>> Loading {model_id} for {len(jobs)} jobs (bs={batch_size}, max_new={max_new_tokens}) ...", flush=True)
    model, tok, stop_ids = load_causal_lm(model_id)
    res = run_loaded_jobs(model_id, model, tok, stop_ids, jobs, batch_size=batch_size, max_new_tokens=max_new_tokens, temperature=temperature)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return res


def load_or_compute(cache_name: str, fn):
    path = os.path.join(OUT_DIR, cache_name)
    if os.path.exists(path) and os.path.getsize(path) > 10:
        print(f"  [CACHE HIT] {cache_name}", flush=True)
        with open(path, "r") as f:
            return json.load(f)
    res = fn()
    atomic_save_json(path, res)
    print(f"  [SAVED] {cache_name} ({len(res)} entries)", flush=True)
    return res


def build_draft_job_specs(parents: dict):
    seen_splits = [("train", parents["train"]), ("val", parents["val"]), ("test_id", parents["test_id"])]
    qwen_draft_jobs, zephyr_draft_jobs = [], []
    draft_meta = {}
    for sp_name, plist in seen_splits:
        for idx, p in enumerate(plist):
            pid = p["parent_id"]
            gen_model = "Qwen/Qwen2.5-7B-Instruct" if (idx % 2 == 0) else "HuggingFaceH4/zephyr-7b-beta"
            tmpl = DRAFT_TEMPLATES_SEEN[idx % len(DRAFT_TEMPLATES_SEEN)]
            twc = max(120, min(250, (p["human_wc"] // 20) * 20))
            up = tmpl.format(prompt=p["prompt"], target_wc=twc)
            draft_meta[pid] = {"split": sp_name, "draft_generator": gen_model}
            jb = {"job_id": pid, "user_prompt": up}
            if gen_model == "Qwen/Qwen2.5-7B-Instruct":
                qwen_draft_jobs.append(jb)
            else:
                zephyr_draft_jobs.append(jb)

    olmo_jobs = []
    for p in parents["test_ood_ar"]:
        twc = max(120, min(250, (p["human_wc"] // 20) * 20))
        up = DRAFT_TEMPLATES_OOD[0].format(prompt=p["prompt"], target_wc=twc)
        olmo_jobs.append({"job_id": p["parent_id"], "user_prompt": up})

    return seen_splits, qwen_draft_jobs, zephyr_draft_jobs, olmo_jobs, draft_meta


def build_seen_edit_specs(seen_splits, all_seen_drafts):
    qwen_edit_jobs, falcon_edit_jobs = [], []
    e4_sum_qwen, e4_sum_falcon = [], []
    edit_assignment = {}
    for sp_name, plist in seen_splits:
        levels = ["E1", "E2", "E3"]
        for idx, p in enumerate(plist):
            pid = p["parent_id"]
            h_txt = p["human_text"]
            m_txt = all_seen_drafts[pid]
            for lv_i, lv in enumerate(levels):
                editor = "Qwen/Qwen2.5-7B-Instruct" if ((idx + lv_i) % 2 == 0) else "tiiuae/Falcon3-7B-Instruct"
                tmpls = EDIT_TEMPLATES_SEEN[lv]
                tmpl = tmpls[(idx * 3 + lv_i) % len(tmpls)]
                edit_assignment[f"{pid}_{lv}"] = editor
                for orig_label, src_txt in [(0, h_txt), (1, m_txt)]:
                    jid = f"{pid}|O{orig_label}|{lv}"
                    jb = {"job_id": jid, "user_prompt": tmpl.format(text=src_txt)}
                    if editor == "Qwen/Qwen2.5-7B-Instruct":
                        qwen_edit_jobs.append(jb)
                    else:
                        falcon_edit_jobs.append(jb)

            if sp_name == "test_id":
                editor = "Qwen/Qwen2.5-7B-Instruct" if (idx % 2 == 0) else "tiiuae/Falcon3-7B-Instruct"
                edit_assignment[f"{pid}_E4"] = editor
                for orig_label, src_txt in [(0, h_txt), (1, m_txt)]:
                    jid = f"{pid}|O{orig_label}|E4_sum"
                    jb = {"job_id": jid, "user_prompt": EDIT_TEMPLATES_SEEN["E4_sum"][0].format(text=src_txt)}
                    if editor == "Qwen/Qwen2.5-7B-Instruct":
                        e4_sum_qwen.append(jb)
                    else:
                        e4_sum_falcon.append(jb)
    return qwen_edit_jobs, falcon_edit_jobs, e4_sum_qwen, e4_sum_falcon, edit_assignment


def gen_llada_all(parents):
    t0 = time.time()
    print("\n>>> Loading GSAI-ML/LLaDA-8B-Instruct (Masked Diffusion LLM) for 150 OOD parents ...", flush=True)
    cfg = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    cfg.use_cache = False
    model_cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], "GSAI-ML/LLaDA-8B-Instruct")
    if not getattr(model_cls, "_tie_patched", False):
        _orig_tie = model_cls.tie_weights
        model_cls.tie_weights = lambda self, *a, **kw: _orig_tie(self)
        model_cls._tie_patched = True
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    model = model_cls.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", config=cfg, dtype=torch.bfloat16).to("cuda").eval()
    out_map = {}
    plist = parents["test_ood_llada"]
    bs = 25
    for st in range(0, len(plist), bs):
        sub = plist[st:st + bs]
        prms = [
            DRAFT_TEMPLATES_OOD[0].format(prompt=p["prompt"], target_wc=max(120, min(220, (p["human_wc"] // 20) * 20)))
            for p in sub
        ]
        gens = llada_generate_batch(model, tok, prms, steps=64, gen_length=256, block_length=64, temperature=0.2)
        for p, g in zip(sub, gens):
            out_map[p["parent_id"]] = g
        print(f"  [LLaDA-8B Diffusion] {min(st + bs, len(plist))}/{len(plist)} ({time.time() - t0:.1f}s)", flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out_map


def run_single_load_editor_bundle(model_id: str, edit_jobs: list, e4_sum_jobs: list, test_id_parents: list, edit_assignment: dict, cache_edits: str, cache_e4: str, batch_size: int):
    p_edits = os.path.join(OUT_DIR, cache_edits)
    p_e4 = os.path.join(OUT_DIR, cache_e4)
    if os.path.exists(p_edits) and os.path.exists(p_e4):
        print(f"  [CACHE HIT] {cache_edits} & {cache_e4}", flush=True)
        return json.load(open(p_edits)), json.load(open(p_e4))

    print(f"\n>>> Single-Load Bundle on {model_id}: edits={len(edit_jobs)}, e4_sum={len(e4_sum_jobs)} (bs={batch_size})", flush=True)
    model, tok, stop_ids = load_causal_lm(model_id)

    if os.path.exists(p_edits):
        edits_out = json.load(open(p_edits))
    else:
        part_main = run_loaded_jobs(model_id, model, tok, stop_ids, edit_jobs, batch_size=batch_size, max_new_tokens=290, temperature=0.5)
        part_sum = run_loaded_jobs(model_id + ":E4_sum", model, tok, stop_ids, e4_sum_jobs, batch_size=batch_size, max_new_tokens=96, temperature=0.4)
        edits_out = {**part_main, **part_sum}
        atomic_save_json(p_edits, edits_out)
        print(f"  [SAVED] {cache_edits} ({len(edits_out)} entries)", flush=True)

    regen_jobs = []
    for p in test_id_parents:
        pid = p["parent_id"]
        if edit_assignment[f"{pid}_E4"] != model_id:
            continue
        twc = max(120, min(240, (p["human_wc"] // 20) * 20))
        for orig_label in [0, 1]:
            summ = edits_out[f"{pid}|O{orig_label}|E4_sum"]
            regen_jobs.append({
                "job_id": f"{pid}|O{orig_label}|E4",
                "user_prompt": EDIT_TEMPLATES_SEEN["E4_regen"][0].format(summary=summ, target_wc=twc),
            })

    e4_out = run_loaded_jobs(model_id + ":E4_regen", model, tok, stop_ids, regen_jobs, batch_size=batch_size, max_new_tokens=280, temperature=0.65)
    atomic_save_json(p_e4, e4_out)
    print(f"  [SAVED] {cache_e4} ({len(e4_out)} entries)", flush=True)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return edits_out, e4_out


def main():
    stage = os.environ.get("CEDAR_STAGE", "all")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs_7b = 144 if vram_gb >= 75 else 80
    bs_phi = 192 if vram_gb >= 75 else 112
    print(f"=== Running step2_generate_symmetric_matrix | stage={stage} | VRAM={vram_gb:.1f}GB | bs_7b={bs_7b} ===", flush=True)

    with open(os.path.join(OUT_DIR, "raw_parents.json")) as f:
        parents = json.load(f)
    with open(os.path.join(OUT_DIR, "external_esl_raw.json")) as f:
        esl_raw = json.load(f)

    seen_splits, qwen_draft_jobs, zephyr_draft_jobs, olmo_jobs, draft_meta = build_draft_job_specs(parents)

    if stage in ("stage2a_g4", "all"):
        load_or_compute("cache_drafts_qwen.json", lambda: run_causal_lm_jobs("Qwen/Qwen2.5-7B-Instruct", qwen_draft_jobs, batch_size=bs_7b, max_new_tokens=290, temperature=0.75))
        load_or_compute("cache_drafts_llada.json", lambda: gen_llada_all(parents))
        if stage == "stage2a_g4":
            return

    if stage in ("stage2a_a100", "all"):
        load_or_compute("cache_drafts_zephyr.json", lambda: run_causal_lm_jobs("HuggingFaceH4/zephyr-7b-beta", zephyr_draft_jobs, batch_size=bs_7b, max_new_tokens=290, temperature=0.75))
        load_or_compute("cache_drafts_olmo2.json", lambda: run_causal_lm_jobs("allenai/OLMo-2-1124-7B-Instruct", olmo_jobs, batch_size=bs_7b, max_new_tokens=290, temperature=0.75))
        if stage == "stage2a_a100":
            return

    # Load all Stage 2A caches
    drafts_qwen = json.load(open(os.path.join(OUT_DIR, "cache_drafts_qwen.json")))
    drafts_zephyr = json.load(open(os.path.join(OUT_DIR, "cache_drafts_zephyr.json")))
    drafts_olmo = json.load(open(os.path.join(OUT_DIR, "cache_drafts_olmo2.json")))
    drafts_llada = json.load(open(os.path.join(OUT_DIR, "cache_drafts_llada.json")))
    all_seen_drafts = {**drafts_qwen, **drafts_zephyr}

    qwen_edit_jobs, falcon_edit_jobs, e4_sum_qwen, e4_sum_falcon, edit_assignment = build_seen_edit_specs(seen_splits, all_seen_drafts)

    if stage in ("stage2b_g4", "all"):
        run_single_load_editor_bundle(
            "Qwen/Qwen2.5-7B-Instruct",
            qwen_edit_jobs,
            e4_sum_qwen,
            parents["test_id"],
            edit_assignment,
            "cache_edits_qwen.json",
            "cache_e4_regen_qwen.json",
            batch_size=bs_7b,
        )
        phi_jobs = []
        for sp_key, draft_dict in [("test_ood_ar", drafts_olmo), ("test_ood_llada", drafts_llada)]:
            for idx, p in enumerate(parents[sp_key]):
                pid = p["parent_id"]
                h_txt = p["human_text"]
                m_txt = draft_dict[pid]
                for lv_i, lv in enumerate(["E1", "E2"]):
                    tmpl = EDIT_TEMPLATES_OOD[lv][(idx + lv_i) % len(EDIT_TEMPLATES_OOD[lv])]
                    for orig_label, src_txt in [(0, h_txt), (1, m_txt)]:
                        phi_jobs.append({"job_id": f"{pid}|O{orig_label}|{lv}", "user_prompt": tmpl.format(text=src_txt)})
        for idx, item in enumerate(esl_raw):
            tmpl = EDIT_TEMPLATES_OOD["E1"][idx % len(EDIT_TEMPLATES_OOD["E1"])]
            phi_jobs.append({"job_id": f"{item['id']}|ESL_E1", "user_prompt": tmpl.format(text=item["text"])})

        load_or_compute(
            "cache_edits_phi4_ood.json",
            lambda: run_causal_lm_jobs("microsoft/Phi-4-mini-instruct", phi_jobs, batch_size=bs_phi, max_new_tokens=290, temperature=0.45),
        )
        if stage == "stage2b_g4":
            return

    if stage in ("stage2b_a100", "all"):
        run_single_load_editor_bundle(
            "tiiuae/Falcon3-7B-Instruct",
            falcon_edit_jobs,
            e4_sum_falcon,
            parents["test_id"],
            edit_assignment,
            "cache_edits_falcon3.json",
            "cache_e4_regen_falcon3.json",
            batch_size=bs_7b,
        )
        if stage == "stage2b_a100":
            return

    # Assemble stage
    edits_qwen = json.load(open(os.path.join(OUT_DIR, "cache_edits_qwen.json")))
    edits_falcon = json.load(open(os.path.join(OUT_DIR, "cache_edits_falcon3.json")))
    e4_qwen_out = json.load(open(os.path.join(OUT_DIR, "cache_e4_regen_qwen.json")))
    e4_falcon_out = json.load(open(os.path.join(OUT_DIR, "cache_e4_regen_falcon3.json")))
    ood_phi_out = json.load(open(os.path.join(OUT_DIR, "cache_edits_phi4_ood.json")))
    seen_edits_all = {**edits_qwen, **edits_falcon, **e4_qwen_out, **e4_falcon_out}

    def token_jaccard_dist(a: str, b: str) -> float:
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa and not sb:
            return 0.0
        return 1.0 - (len(sa & sb) / max(1, len(sa | sb)))

    NOMINAL_STRENGTH = {"E0": 0.0, "E1": 0.2, "E2": 0.5, "E3": 0.8, "E4": 1.0}
    assembled = {"train": [], "val": [], "test_id": [], "test_ood_ar": [], "test_ood_llada": []}

    for sp_name, plist in seen_splits:
        lvs = ["E0", "E1", "E2", "E3"] + (["E4"] if sp_name == "test_id" else [])
        for p in plist:
            pid = p["parent_id"]
            d_gen = draft_meta[pid]["draft_generator"]
            base_texts = {0: p["human_text"], 1: all_seen_drafts[pid]}
            for orig_label in [0, 1]:
                raw_t = base_texts[orig_label]
                for lv in lvs:
                    txt = raw_t if lv == "E0" else seen_edits_all[f"{pid}|O{orig_label}|{lv}"]
                    ed_mod = "none" if lv == "E0" else edit_assignment[f"{pid}_{lv}"]
                    assembled[sp_name].append({
                        "id": f"{pid}_O{orig_label}_{lv}",
                        "parent_id": pid,
                        "split": sp_name,
                        "domain": p["domain"],
                        "subdomain": p["subdomain"],
                        "origin_label": orig_label,
                        "edit_level": lv,
                        "nominal_strength": NOMINAL_STRENGTH[lv],
                        "measured_lex_dist": round(token_jaccard_dist(raw_t, txt), 4),
                        "draft_generator": "human" if orig_label == 0 else d_gen,
                        "editor_model": ed_mod,
                        "text": txt,
                        "wc": len(txt.split()),
                    })

    for sp_key, draft_dict, d_gen_name in [
        ("test_ood_ar", drafts_olmo, "allenai/OLMo-2-1124-7B-Instruct"),
        ("test_ood_llada", drafts_llada, "GSAI-ML/LLaDA-8B-Instruct"),
    ]:
        for p in parents[sp_key]:
            pid = p["parent_id"]
            base_texts = {0: p["human_text"], 1: draft_dict[pid]}
            for orig_label in [0, 1]:
                raw_t = base_texts[orig_label]
                for lv in ["E0", "E1", "E2"]:
                    txt = raw_t if lv == "E0" else ood_phi_out[f"{pid}|O{orig_label}|{lv}"]
                    assembled[sp_key].append({
                        "id": f"{pid}_O{orig_label}_{lv}",
                        "parent_id": pid,
                        "split": sp_key,
                        "domain": p["domain"],
                        "subdomain": p["subdomain"],
                        "origin_label": orig_label,
                        "edit_level": lv,
                        "nominal_strength": NOMINAL_STRENGTH[lv],
                        "measured_lex_dist": round(token_jaccard_dist(raw_t, txt), 4),
                        "draft_generator": "human" if orig_label == 0 else d_gen_name,
                        "editor_model": "none" if lv == "E0" else "microsoft/Phi-4-mini-instruct",
                        "text": txt,
                        "wc": len(txt.split()),
                    })

    for k, rows in assembled.items():
        out_p = os.path.join(OUT_DIR, f"split_{k}.json")
        atomic_save_json(out_p, rows)
        print(f"Saved split_{k}.json: {len(rows)} rows", flush=True)

    esl_full = []
    for item in esl_raw:
        raw_t = item["text"]
        pol_t = ood_phi_out[f"{item['id']}|ESL_E1"]
        esl_full.append({**item, "edit_level": "E0", "subset": "esl_unedited", "measured_lex_dist": 0.0})
        esl_full.append({
            **item,
            "id": f"{item['id']}_E1",
            "text": pol_t,
            "wc": len(pol_t.split()),
            "edit_level": "E1",
            "subset": "esl_grammar_polished",
            "measured_lex_dist": round(token_jaccard_dist(raw_t, pol_t), 4),
        })
    atomic_save_json(os.path.join(OUT_DIR, "external_esl_full.json"), esl_full)
    print(f"Saved external_esl_full.json: {len(esl_full)} rows (300 E0 + 300 E1)", flush=True)
    print("\n=== ALL SYMMETRIC INTERVENTION SPLITS COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
