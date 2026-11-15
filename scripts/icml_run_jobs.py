#!/usr/bin/env python3
"""Run lists of P1-style jobs on Colab G4 sessions. PLAN json: {session: [{env...}, ...]}.
Each session: new -> upload bundle -> for each job: exec trainer, download result with retries -> stop (verified).
Results to results_icml/<OUTDIR> (default p1). Usage: python scripts/icml_run_jobs.py PLAN.json [OUTDIR]
A job may set "TRAINER" (default icml_p1_llm_pair.py); icml_final_train.py also writes <tag>_seed<seed>_weights.pt,
which is fetched (best effort) to results_icml/<OUTDIR>/weights."""
import json, os, subprocess, sys, tarfile, threading, time
COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_icml", sys.argv[2] if len(sys.argv) > 2 else "p1"); LOG = os.path.join(ROOT, "logs", "icml")
os.makedirs(os.path.join(OUT, "weights"), exist_ok=True); os.makedirs(LOG, exist_ok=True)
PLAN = json.load(open(sys.argv[1])); lock = threading.Lock()
def say(m):
    with lock: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def run(a, t):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=t)
    except Exception as e: say(f"WARN {a[1:4]} {e}"); return None
def active(sess):
    r = run([COLAB, "sessions"], 60); return bool(r and f"[{sess}]" in r.stdout)
def handle(sess, jobs, bp, up):
    try:
        ok = False
        for attempt in range(4):
            if not active(sess):
                r = run([COLAB, "new", "--gpu", "A100" if "a100" in sess else "G4", "-s", sess], 600)  # session name picks the GPU; say(f"{sess} new rc={r.returncode if r else None}")
                time.sleep(10)
            run([COLAB, "upload", "-s", sess, bp, "/content/p1_bundle.tar.gz"], 600)
            ex = run([COLAB, "exec", "-s", sess, "-f", up, "--timeout", "300"], 400)
            if ex and "P1_UNPACK_OK" in ex.stdout:
                ok = True; break
            say(f"{sess} unpack attempt {attempt} failed; retrying"); time.sleep(30)
        if not ok: say(f"{sess} unpack failed"); return
        for env in jobs:
            fn = f"{env['P1_TAG']}_seed{env['P1_SEED']}.json"; dst = os.path.join(OUT, fn)
            if os.path.exists(dst): say(f"{sess} skip {fn}"); continue
            cmd = [COLAB, "exec", "-s", sess]
            env = dict(env); trainer = env.pop("TRAINER", "icml_p1_llm_pair.py")
            full = {"P1_PAIR_W": "0.5", "P1_EPOCHS": "2", "P1_LORA_R": "16", "P1_LR": "1e-4",
                    "P1_LOSS": "pair", "P1_AUX_W": "0.5", "P1_MAX_STEPS": "0"}; full.update(env)
            for k, v in full.items(): cmd += ["--env", f"{k}={v}"]   # pass every var: kernel env persists across execs
            cmd += ["-f", os.path.join(ROOT, "scripts", trainer), "--timeout", "12000"]
            with open(os.path.join(LOG, f"{sess}.log"), "a") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=12600).returncode
            # the exec stream can drop (websocket) while the kernel keeps training: wait for the file itself
            for i in range(90):
                ls = run([COLAB, "ls", "-s", sess, "/content/p1_out"], 90)
                if ls and fn in ls.stdout: break
                time.sleep(60)
            for i in range(8):
                run([COLAB, "download", "-s", sess, f"/content/p1_out/{fn}", dst], 240)
                if os.path.exists(dst) and os.path.getsize(dst) > 1000: break
                time.sleep(20)
            if trainer == "icml_final_train.py":
                wf = fn.replace(".json", "_weights.pt"); wd = os.path.join(OUT, "weights", wf)
                for i in range(3):
                    run([COLAB, "download", "-s", sess, f"/content/p1_out/{wf}", wd], 900)
                    if os.path.exists(wd) and os.path.getsize(wd) > 1e6: break
                    time.sleep(20)
            say(f"{sess} {fn} rc={rc} fetched={os.path.exists(dst)}")
    finally:
        for i in range(4):
            run([COLAB, "stop", "-s", sess], 120); time.sleep(5)
            if not active(sess): break
        say(f"STOPPED {sess} verified={not active(sess)}")
if __name__ == "__main__":
    tag = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    bp = os.path.join(LOG, f"bundle_{tag}.tar.gz"); up = os.path.join(LOG, f"unpack_{tag}.py")
    with tarfile.open(bp, "w:gz") as tf:
        tf.add(os.path.join(ROOT, "scripts/icml_p1_llm_pair.py"), arcname="icml_p1_llm_pair.py")
        tf.add(os.path.join(ROOT, "scripts/icml_final_train.py"), arcname="icml_final_train.py")
        tf.add(os.path.join(ROOT, "scripts/v3_eval_pretrained_on_confirm.py"), arcname="scripts/v3_eval_pretrained_on_confirm.py")
        tf.add(os.path.join(ROOT, "data_v2/provenance_lock.json"), arcname="cedar_data_v2/provenance_lock.json")
        for rel, arc in [("data/split_val.json", "cedar_data"), ("data/split_test_id.json", "cedar_data"), ("data/split_test_ood_ar.json", "cedar_data"), ("data/split_test_ood_llada.json", "cedar_data"),
                         ("data/external_esl_full.json", "cedar_data"), ("data/external_mixset.json", "cedar_data"), ("data/external_beemo.json", "cedar_data"),
                         ("data/external_mage_wild.json", "cedar_data"),
                         ("data_v2/split_train_v2.json", "cedar_data_v2"), ("data_v2/val_manifest.json", "cedar_data_v2"),
                         ("data_v2/split_test_confirm.json", "cedar_data_v2"), ("data_e2/split_val_shift.json", "cedar_data_e2"),
                         ("data_e2/split_test_confirm2.json", "cedar_data_e2"), ("data_c3/split_test_confirm3.json", "cedar_data_c3")]:
            if os.path.exists(os.path.join(ROOT, rel)):
                tf.add(os.path.join(ROOT, rel), arcname=f"{arc}/{os.path.basename(rel)}")
    open(up, "w").write("import tarfile,os\ntarfile.open('/content/p1_bundle.tar.gz').extractall('/content')\nprint('P1_UNPACK_OK')\n")
    th = [threading.Thread(target=handle, args=(s, j, bp, up)) for s, j in PLAN.items()]
    for t in th: t.start(); time.sleep(5)
    for t in th: t.join()
    u = run([COLAB, "usage"], 60); print(u.stdout if u else ""); say("JOBS DONE")
