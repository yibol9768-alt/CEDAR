#!/usr/bin/env python3
"""Drive one Confirm-3 generation part on a Colab session: smoke (4 jobs/group) -> download -> full -> download -> stop.
Usage: python scripts/icml_run_c3.py PART(a|b) SESSION GPU   (session is created if absent; always stopped at the end)"""
import os, subprocess, sys, tarfile, time
COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C3 = os.path.join(ROOT, "data_c3"); LOG = os.path.join(ROOT, "logs", "icml")
PART, SESS, GPU = sys.argv[1], sys.argv[2], sys.argv[3]


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(a, t):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=t)
    except Exception as e:
        say(f"WARN {a[1:4]} {e}"); return None


def active():
    r = run([COLAB, "sessions"], 60); return bool(r and f"[{SESS}]" in r.stdout)


def fetch(names):
    for fn in names:
        dst = os.path.join(C3, fn)
        for i in range(8):
            run([COLAB, "download", "-s", SESS, f"/content/cedar_data_c3/{fn}", dst], 300)
            if os.path.exists(dst) and os.path.getsize(dst) > 200:
                break
            time.sleep(20)
        say(f"fetched {fn}: {os.path.exists(dst)}")


def gen(smoke, launch=True):
    env = {"CEDAR_C3_STAGE": f"part_{PART}", "CEDAR_C3_SMOKE": "4" if smoke else "0", "CEDAR_C3_DIR": "/content/cedar_data_c3"}
    cmd = [COLAB, "exec", "-s", SESS]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd += ["-f", os.path.join(ROOT, "scripts/icml_build_confirm3.py"), "--timeout", "14000"]
    rc = None
    if launch:
        with open(os.path.join(LOG, f"{SESS}_{PART}.log"), "a") as lf:
            lf.write(f"\n===== {'SMOKE' if smoke else 'FULL'} {time.strftime('%H:%M:%S')}\n"); lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=14400).returncode
    suf = "_smoke" if smoke else ""
    gi = 0 if PART == "a" else 1
    names = [f"c3_drafts_g{gi}{suf}.json", f"c3_edits_{PART}{suf}.json"]
    for i in range(120):  # the exec stream can drop while the kernel keeps generating
        ls = run([COLAB, "ls", "-s", SESS, "/content/cedar_data_c3"], 90)
        if ls and all(n in ls.stdout for n in names):
            break
        if "ipykernel_" in open(os.path.join(LOG, f"{SESS}_{PART}.log")).read().split("=====")[-1]:  # kernel-side error only
            raise SystemExit("generation failed; see log")
        time.sleep(60)
    say(f"{'smoke' if smoke else 'full'} rc={rc}")
    fetch(names)


RESUME = os.environ.get("C3_RESUME", "")  # "smoke_wait": smoke exec already running remotely; "full": skip smoke
try:
    if not active():
        r = run([COLAB, "new", "--gpu", GPU, "-s", SESS], 600); say(f"new rc={r.returncode if r else None}")
    bp = os.path.join(LOG, f"bundle_c3_{PART}.tar.gz"); up = os.path.join(LOG, "unpack_c3.py")
    with tarfile.open(bp, "w:gz") as tf:
        for rel, arc in [("scripts/icml_build_confirm3.py", "icml_build_confirm3.py"),
                         ("scripts/v3_build_e2_splits.py", "v3_build_e2_splits.py"),
                         ("scripts/v3_build_v2_benchmark.py", "v3_build_v2_benchmark.py"),
                         ("data_c3/parents_and_schedule_c3.json", "cedar_data_c3/parents_and_schedule_c3.json"),
                         ("data_v2/provenance_lock.json", "cedar_data_v2/provenance_lock.json")]:
            tf.add(os.path.join(ROOT, rel), arcname=arc)
    open(up, "w").write("import tarfile,os\ntarfile.open('/content/c3_bundle.tar.gz').extractall('/content')\n"
                        "print('C3_UNPACK_OK', sorted(os.listdir('/content/cedar_data_c3')))\n")
    for attempt in range(0 if RESUME else 5):
        run([COLAB, "upload", "-s", SESS, bp, "/content/c3_bundle.tar.gz"], 600)
        ex = run([COLAB, "exec", "-s", SESS, "-f", up, "--timeout", "300"], 400)
        ls = run([COLAB, "ls", "-s", SESS, "/content"], 90)
        say(f"unpack attempt {attempt}: {ex.stdout.strip()[-200:] if ex else None}")
        if (ex and "C3_UNPACK_OK" in ex.stdout) or (ls and "icml_build_confirm3.py" in ls.stdout):
            break
        time.sleep(30)
    else:
        if not RESUME:
            raise SystemExit("unpack failed")
    if RESUME != "full":
        gen(smoke=True, launch=(RESUME != "smoke_wait"))
    gen(smoke=False)
finally:
    for i in range(4):
        run([COLAB, "stop", "-s", SESS], 120); time.sleep(5)
        if not active():
            break
    say(f"STOPPED {SESS} verified={not active()}")
    say("C3 PART DONE")
