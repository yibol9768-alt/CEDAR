#!/usr/bin/env python3
"""Drive Confirm-3G re-scoring on ONE Colab session: new -> upload bundle + weights -> exec icml_c3g_rescore.py
-> wait for the result file -> download -> stop (verified, always). Usage: python scripts/icml_run_c3g.py SESSION [GPU]"""
import glob, os, subprocess, sys, tarfile, time
COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "icml"); OUTD = os.path.join(ROOT, "results_icml", "c3g"); os.makedirs(OUTD, exist_ok=True)
SESS = sys.argv[1]; GPU = sys.argv[2] if len(sys.argv) > 2 else "G4"
FN = "c3g_rescore_seed0.json"
PROBE = os.path.join(LOG, "probe_c3g.py")


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(a, t):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=t)
    except Exception as e:  # noqa: BLE001
        say(f"WARN {a[1:4]} {e}"); return None


def active():
    r = run([COLAB, "sessions"], 60); return bool(r and f"[{SESS}]" in r.stdout)


def upload(src, dst, minsize):
    for i in range(5):
        run([COLAB, "upload", "-s", SESS, src, dst], 1200)
        ls = run([COLAB, "ls", "-s", SESS, os.path.dirname(dst)], 90)
        if ls and os.path.basename(dst) in ls.stdout:
            return True
        time.sleep(20)
    raise SystemExit(f"upload failed: {src}")


def main():
    bp = os.path.join(LOG, "bundle_c3g.tar.gz"); up = os.path.join(LOG, "unpack_c3g.py")
    with tarfile.open(bp, "w:gz") as tf:
        tf.add(os.path.join(ROOT, "scripts/v3_eval_pretrained_on_confirm.py"), arcname="scripts/v3_eval_pretrained_on_confirm.py")
        tf.add(os.path.join(ROOT, "data_v2/provenance_lock.json"), arcname="cedar_data_v2/provenance_lock.json")
        for rel, arc in [("data/split_val.json", "cedar_data"), ("data_v2/val_manifest.json", "cedar_data_v2"),
                         ("data_c3/split_test_confirm3.json", "cedar_data_c3"), ("data_c3g/split_test_confirm3g.json", "cedar_data_c3g")]:
            tf.add(os.path.join(ROOT, rel), arcname=f"{arc}/{os.path.basename(rel)}")
        for w in sorted(glob.glob(os.path.join(ROOT, "results_icml/final/weights/f_*_weights.pt"))):
            tf.add(w, arcname=f"weights/{os.path.basename(w)}")
    open(up, "w").write("import tarfile\ntarfile.open('/content/c3g_bundle.tar.gz').extractall('/content')\n"
                        "import glob;print('C3G_UNPACK_OK', len(glob.glob('/content/weights/f_*.pt')))\n")
    open(PROBE, "w").write("import json\ntry:\n    d=json.load(open('/content/p1_out/c3g_rescore_seed0.json'));print('PROBE',len(d['runs']),bool(d['pretrained']))\n"
                           "except Exception as e:\n    print('PROBE_ERR',e)\n")
    say(f"bundle {os.path.getsize(bp) / 1e9:.2f} GB")
    try:
        for attempt in range(3):
            if not active():
                r = run([COLAB, "new", "--gpu", GPU, "-s", SESS], 600); say(f"new rc={r.returncode if r else None}"); time.sleep(10)
            upload(bp, "/content/c3g_bundle.tar.gz", 1e9)
            ex = run([COLAB, "exec", "-s", SESS, "-f", up, "--timeout", "900"], 1000)
            if ex and "C3G_UNPACK_OK" in ex.stdout:
                say(ex.stdout.strip().splitlines()[-1]); break
            say(f"unpack attempt {attempt} failed")
        else:
            raise SystemExit("unpack failed")
        with open(os.path.join(LOG, f"{SESS}.log"), "a") as lf:
            rc = subprocess.run([COLAB, "exec", "-s", SESS, "-f", os.path.join(ROOT, "scripts/icml_c3g_rescore.py"), "--timeout", "10000"],
                                stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=10800).returncode
        say(f"exec rc={rc}")
        for i in range(120):  # the exec stream can drop while the kernel keeps working
            if "C3G_RESCORE_DONE" in open(os.path.join(LOG, f"{SESS}.log")).read():
                break
            probe = run([COLAB, "exec", "-s", SESS, "--timeout", "60", "-f", PROBE], 120)
            if probe and "PROBE 25 True" in probe.stdout:
                break
            time.sleep(60)
        dst = os.path.join(OUTD, FN)
        for i in range(8):
            run([COLAB, "download", "-s", SESS, f"/content/p1_out/{FN}", dst], 600)
            if os.path.exists(dst) and os.path.getsize(dst) > 1e5:
                break
            time.sleep(20)
        say(f"fetched {os.path.exists(dst)}")
    finally:
        for i in range(4):
            run([COLAB, "stop", "-s", SESS], 120); time.sleep(5)
            if not active():
                break
        say(f"STOPPED {SESS} verified={not active()}")


if __name__ == "__main__":
    main()
