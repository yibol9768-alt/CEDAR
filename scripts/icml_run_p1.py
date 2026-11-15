#!/usr/bin/env python3
"""Launch P1 pilots on Colab G4 (one session per model), fetch predictions to results_icml/p1, stop sessions.
Usage: python scripts/icml_run_p1.py  (edits JOBS below). Only stops its own cedar_p1* sessions."""
import os, subprocess, sys, tarfile, threading, time
COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_icml", "p1"); LOG = os.path.join(ROOT, "logs", "icml"); os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)
JOBS = {"cedar_p1a": {"P1_MODEL": "Qwen/Qwen3-1.7B-Base", "P1_TAG": "qwen3_1p7b_pair", "P1_SEED": "42"},
        "cedar_p1b": {"P1_MODEL": "Qwen/Qwen3-8B-Base", "P1_TAG": "qwen3_8b_pair", "P1_SEED": "42"}}
lock = threading.Lock()
def say(m):
    with lock: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def run(a, t):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=t)
    except Exception as e: say(f"WARN {a[1:4]} {e}"); return None
def bundle():
    p = os.path.join(LOG, "p1_bundle.tar.gz")
    with tarfile.open(p, "w:gz") as tf:
        tf.add(os.path.join(ROOT, "scripts/icml_p1_llm_pair.py"), arcname="icml_p1_llm_pair.py")
        for rel, arc in [("data/split_val.json", "cedar_data"), ("data/split_test_id.json", "cedar_data"),
                         ("data_v2/split_train_v2.json", "cedar_data_v2"), ("data_v2/val_manifest.json", "cedar_data_v2"),
                         ("data_v2/split_test_confirm.json", "cedar_data_v2"), ("data_e2/split_val_shift.json", "cedar_data_e2"),
                         ("data_e2/split_test_confirm2.json", "cedar_data_e2")]:
            tf.add(os.path.join(ROOT, rel), arcname=f"{arc}/{os.path.basename(rel)}")
    u = os.path.join(LOG, "p1_unpack.py")
    open(u, "w").write("import tarfile,shutil,os\ntarfile.open('/content/p1_bundle.tar.gz').extractall('/content')\nprint('P1_UNPACK_OK', os.listdir('/content'))\n")
    return p, u
def handle(sess, env, bp, up):
    r = run([COLAB, "new", "--gpu", "G4", "-s", sess], 600); say(f"{sess} new rc={r.returncode if r else None}")
    try:
        u1 = run([COLAB, "upload", "-s", sess, bp, "/content/p1_bundle.tar.gz"], 600)
        ex = run([COLAB, "exec", "-s", sess, "-f", up, "--timeout", "300"], 400)
        if not (ex and "P1_UNPACK_OK" in ex.stdout): say(f"{sess} unpack failed {ex.stdout if ex else ''} {ex.stderr if ex else ''}"); return
        cmd = [COLAB, "exec", "-s", sess]
        for k, v in env.items(): cmd += ["--env", f"{k}={v}"]
        cmd += ["-f", os.path.join(ROOT, "scripts/icml_p1_llm_pair.py"), "--timeout", "12000"]
        with open(os.path.join(LOG, f"{sess}.log"), "a") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=12600).returncode
        say(f"{sess} worker rc={rc}")
        fn = f"{env['P1_TAG']}_seed{env['P1_SEED']}.json"
        d = run([COLAB, "download", "-s", sess, f"/content/p1_out/{fn}", os.path.join(OUT, fn)], 300)
        say(f"{sess} download rc={d.returncode if d else None}")
    finally:
        run([COLAB, "stop", "-s", sess], 120); say(f"STOPPED {sess}")
if __name__ == "__main__":
    bp, up = bundle()
    th = [threading.Thread(target=handle, args=(s, e, bp, up)) for s, e in JOBS.items()]
    for t in th: t.start(); time.sleep(5)
    for t in th: t.join()
    u = run([COLAB, "usage"], 60); print(u.stdout if u else "")
    say("P1 DONE")
