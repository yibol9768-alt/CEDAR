"""E2 driver: python scripts/v3_run_e2.py PLAN.json  (PLAN = {"session_name": ["run_seedN", ...], ...}).
Opens one G4 per session, uploads code + data (+ whatever data_e2 splits exist), runs the named jobs,
syncs predictions to results_e2/model_preds, and stops each session as soon as its own jobs are synced."""
import os, re, sys, glob, time, tarfile, subprocess, threading

COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.environ.get("ICML_PRED") or os.path.join(ROOT, "results_e2", "model_preds")  # ICML final reruns: results_icml/final_enc
os.makedirs(PRED, exist_ok=True)
LOG = os.environ.get("ICML_LOG") or os.path.join(ROOT, "logs", "e2")
os.makedirs(LOG, exist_ok=True)
JSON_RE = re.compile(r"^[a-zA-Z0-9_]+\.json$")
lock = threading.Lock()
import json
PLAN = json.load(open(sys.argv[1]))
S = list(PLAN)


def say(m):
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(args, timeout):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        say(f"WARN {args[1:4]} {e}")
        return None


def bundle():
    path = os.path.join(LOG, "bundle.tar.gz")
    with tarfile.open(path, "w:gz") as tf:
        for rel in ["src/cedar_model.py", "src/cedar_v2_model.py", "scripts/v3_train_cedar_v2_suite.py", "scripts/v3_run_named_jobs.py"]:
            tf.add(os.path.join(ROOT, rel), arcname=rel)
        for df in sorted(glob.glob(os.path.join(ROOT, "data", "*.json"))):
            tf.add(df, arcname=f"cedar_data/{os.path.basename(df)}")
        for df in sorted(glob.glob(os.path.join(ROOT, "data_v2", "*.json"))):
            tf.add(df, arcname=f"cedar_data_v2/{os.path.basename(df)}")
        for sp in ("val_shift", "test_confirm2"):
            fp = os.path.join(ROOT, "data_e2", f"split_{sp}.json")
            if os.path.exists(fp) and (sp != "test_confirm2" or os.environ.get("E2_WITH_CONFIRM2") == "1"):
                tf.add(fp, arcname=f"cedar_data_e2/{os.path.basename(fp)}")
        c3 = os.path.join(ROOT, "data_c3", "split_test_confirm3.json")  # ICML Confirm-3 (sealed)
        if os.path.exists(c3) and os.environ.get("E2_WITH_CONFIRM3") == "1":
            tf.add(c3, arcname="cedar_data_c3/split_test_confirm3.json")
    unpack = os.path.join(LOG, "unpack.py")
    open(unpack, "w").write(
        "import os, shutil, tarfile, subprocess, sys\n"
        "for pkg in ['sentencepiece', 'tiktoken']:\n"
        "    try:\n        __import__(pkg)\n"
        "    except ImportError:\n        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])\n"
        "tarfile.open('/content/e2_bundle.tar.gz').extractall('/tmp/e2x')\n"
        "for s_d, d_d in [('src','/content/src'),('scripts','/content/scripts'),('cedar_data','/content/cedar_data'),('cedar_data_v2','/content/cedar_data_v2'),('cedar_data_e2','/content/cedar_data_e2'),('cedar_data_c3','/content/cedar_data_c3')]:\n"
        "    if os.path.isdir(f'/tmp/e2x/{s_d}'): shutil.copytree(f'/tmp/e2x/{s_d}', d_d, dirs_exist_ok=True)\n"
        "for p in ['/content/src/__init__.py','/content/scripts/__init__.py']:\n"
        "    open(p,'a').close()\n"
        "os.makedirs('/content/cedar_results_v3/model_preds_v3', exist_ok=True)\n"
        "os.makedirs('/content/cedar_results_v3/checkpoints', exist_ok=True)\n"
        "print('E2_UNPACK_OK')\n")
    return path, unpack


def sync(sess):
    r = run([COLAB, "ls", "-s", sess, "/content/cedar_results_v3/model_preds_v3"], 90)
    if not r or r.returncode != 0:
        return None
    remote = [l.strip() for l in r.stdout.splitlines() if JSON_RE.match(l.strip())]
    for fn in remote:
        dst = os.path.join(PRED, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        part = dst + ".part"
        d = run([COLAB, "download", "-s", sess, f"/content/cedar_results_v3/model_preds_v3/{fn}", part], 180)
        if d and d.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > 1000:
            os.replace(part, dst)
            say(f"{sess} synced {fn}")
        elif os.path.exists(part):
            os.remove(part)
    return [fn for fn in remote if not os.path.exists(os.path.join(PRED, fn))]


def handle(sess, jobs, bpath, upath):
    r = run([COLAB, "new", "--gpu", "A100" if "a100" in sess else "G4", "-s", sess], 600)  # session name picks the GPU
    say(f"{sess} new rc={r.returncode if r else None}")
    up = run([COLAB, "upload", "-s", sess, bpath, "/content/e2_bundle.tar.gz"], 600)
    ex = run([COLAB, "exec", "-s", sess, "-f", upath, "--timeout", "300"], 400)
    ok = bool(up and up.returncode == 0 and ex and "E2_UNPACK_OK" in ex.stdout)
    say(f"{sess} unpack ok={ok}")
    if ok:
        stop_evt = threading.Event()

        def keep_syncing():
            while not stop_evt.wait(120):
                sync(sess)
        t = threading.Thread(target=keep_syncing, daemon=True)
        t.start()
        cmd = [COLAB, "exec", "-s", sess, "--env", f"CEDAR_JOBS={','.join(jobs)}", "-f",
               os.path.join(ROOT, "scripts", "v3_run_named_jobs.py"), "--timeout", "14000"]
        with open(os.path.join(LOG, f"{sess}.log"), "a") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=14400).returncode
        stop_evt.set()
        say(f"{sess} worker rc={rc}")
    clean = False
    for _ in range(6):
        miss = sync(sess)
        if miss == []:
            clean = True
            break
        time.sleep(20)
    want = {f"{j}.json" for j in jobs}
    have = {os.path.basename(f) for f in glob.glob(os.path.join(PRED, "*.json"))}
    say(f"{sess} sync clean={clean} missing_jobs={sorted(want - have)}")
    run([COLAB, "stop", "-s", sess], 120)
    say(f"STOPPED {sess}")


if __name__ == "__main__":
    bpath, upath = bundle()
    ths = [threading.Thread(target=handle, args=(s, PLAN[s], bpath, upath)) for s in S]
    for t in ths:
        t.start()
        time.sleep(5)
    for t in ths:
        t.join()
    u = run([COLAB, "usage"], 60)
    print(u.stdout if u else "", flush=True)
    say("E2 DONE")
