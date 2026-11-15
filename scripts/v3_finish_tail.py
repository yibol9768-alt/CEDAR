"""Tail driver after the 4-way orchestrator was stopped (f3 A100 OOM on DeBERTa).
Per session: wait for its original worker to exit, sync its JSONs, optionally run extra named jobs,
sync again, then stop the session. f3 also runs the Theorem-1 geometry (it holds the seed42 ckpt)."""
import os, re, sys, time, subprocess, threading

COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results_v3", "model_preds_v3")
LOG = os.path.join(ROOT, "logs", "colab_workers")
JSON_RE = re.compile(r"^[a-zA-Z0-9_]+\.json$")
lock = threading.Lock()


def say(msg):
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(args, timeout):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        say(f"WARN {args[1:4]} {e}")
        return None


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


def sync_until_clean(sess, tries=6):
    for _ in range(tries):
        miss = sync(sess)
        if miss == []:
            return True
        time.sleep(20)
    return False


def exec_script(sess, script, envs, timeout):
    cmd = [COLAB, "exec", "-s", sess]
    for e in envs:
        cmd += ["--env", e]
    cmd += ["-f", os.path.join(ROOT, "scripts", script), "--timeout", str(timeout)]
    with open(os.path.join(LOG, f"{sess}.log"), "a") as lf:
        lf.write(f"\n=== TAIL SPAWN {time.strftime('%H:%M:%S')} {sess} {script} {envs} ===\n")
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=timeout + 120).returncode
    say(f"{sess} {script} {envs} rc={rc}")
    return rc


def handle(sess, wait_pid, jobs, force_gc, theorem):
    if wait_pid:
        while os.path.exists(f"/proc/{wait_pid}"):
            time.sleep(15)
        say(f"{sess} original worker exited")
    ok = sync_until_clean(sess)
    say(f"{sess} first sync clean={ok}")
    if jobs or theorem:
        run([COLAB, "restart-kernel", "-s", sess], 120)
    if jobs:
        envs = [f"CEDAR_JOBS={','.join(jobs)}"] + (["CEDAR_FORCE_GC=1"] if force_gc else [])
        for attempt in range(2):
            if exec_script(sess, "v3_run_named_jobs.py", envs, 5400) == 0:
                break
            run([COLAB, "restart-kernel", "-s", sess], 120)
        ok = sync_until_clean(sess)
        say(f"{sess} post-jobs sync clean={ok}")
    if theorem:
        run([COLAB, "restart-kernel", "-s", sess], 120)
        rc = exec_script(sess, "v3_theorem2_convergence_and_geometry.py", [], 2400)
        dst = os.path.join(ROOT, "results_v3", "nsd_v3_theorem2_and_geometry.json")
        d = run([COLAB, "download", "-s", sess, "/content/cedar_results_v3/nsd_v3_theorem2_and_geometry.json", dst], 180)
        say(f"{sess} theorem rc={rc} downloaded={bool(d and d.returncode == 0 and os.path.exists(dst))}")
    if ok:
        run([COLAB, "stop", "-s", sess], 120)
        say(f"STOPPED {sess}")
    else:
        say(f"NOT STOPPED {sess}: files still missing locally")


PLAN = [
    # sess, original exec pid, extra jobs, force_gc, theorem
    ("cedar_v3_f3", None, ["db_v3_unedited_erm_seed3042"], True, True),
    ("cedar_v3_f1", 921801, ["db_v3_augmented_erm_seed2042"], False, False),
    ("cedar_v3_f2", 921802, ["db_v3_erm_paired_consist_seed3042"], False, False),
    ("cedar_v3_f0", 921800, [], False, False),
]

if __name__ == "__main__":
    ths = [threading.Thread(target=handle, args=p) for p in PLAN]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    n = len([f for f in os.listdir(PRED) if f.endswith(".json")])
    say(f"ALL DONE local JSONs={n}/116")
    u = run([COLAB, "usage"], 60)
    print(u.stdout if u else "", flush=True)
